from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding, TokenEmbedding, ReplicationPad1d
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FrequencyAwarePatchBlock(nn.Module):
    """
    Learnable Frequency-Aware Multi-Scale Patching Block.
    
    Uses FFT to analyze frequency content and attention mechanism to learn
    optimal scale/patch length combinations dynamically for each input.
    
    Key Features:
    - FFT-based frequency analysis to understand input periodicity
    - Learnable scale queries that attend to frequency representations
    - Adaptive weighting of multiple patch scales based on input characteristics
    - End-to-end differentiable for joint optimization
    """
    
    def __init__(self, configs, candidate_patch_lens=None, dropout=0.1):
        """
        Args:
            configs: Model configuration object
            candidate_patch_lens: List of candidate patch lengths to consider.
                                  If None, uses adaptive defaults based on seq_len.
            dropout: Dropout rate
        """
        super(FrequencyAwarePatchBlock, self).__init__()
        
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        
        # Auto-compute candidate patch lengths based on sequence length if not provided
        if candidate_patch_lens is None:
            # Generate patch lengths that cover different frequency ranges
            # Typically: very short (high freq), short, medium, long (low freq)
            base = max(4, self.seq_len // 24)
            candidate_patch_lens = [
                base,                          # High frequency capture
                base * 2,                      # Medium-high frequency
                base * 4,                      # Medium frequency
                min(base * 8, self.seq_len // 2),  # Low frequency
            ]
            # Remove duplicates and sort
            candidate_patch_lens = sorted(list(set(candidate_patch_lens)))
        
        self.candidate_patch_lens = candidate_patch_lens
        self.n_scales = len(candidate_patch_lens)
        
        # Compute strides (50% overlap for each scale)
        self.strides = [max(1, pl // 2) for pl in candidate_patch_lens]
        
        # Compute number of patches per scale
        self.patch_nums_list = [
            int((self.seq_len - pl) / st + 2)
            for pl, st in zip(self.candidate_patch_lens, self.strides)
        ]
        self.total_patch_nums = sum(self.patch_nums_list)
        
        # ========== Frequency Analysis Module ==========
        self.freq_dim = self.seq_len // 2 + 1  # Number of frequency bins from rfft
        
        # Frequency magnitude encoder
        self.freq_encoder = nn.Sequential(
            nn.Linear(self.freq_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # Frequency phase encoder (phase contains temporal info)
        self.phase_encoder = nn.Sequential(
            nn.Linear(self.freq_dim, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, self.d_model)
        )
        
        # Combine magnitude and phase
        self.freq_fusion = nn.Linear(self.d_model * 2, self.d_model)
        
        # ========== Learnable Scale Selection Module ==========
        # Learnable scale queries - each represents a patch length candidate
        self.scale_queries = nn.Parameter(
            torch.randn(self.n_scales, self.d_model) * 0.02
        )
        
        # Cross-attention: scale queries attend to frequency content
        self.scale_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=configs.n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Scale weight projection (outputs soft attention over scales)
        self.scale_weight_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, 1)
        )
        
        # Learnable prior bias for each scale (can learn dataset-specific preferences)
        self.scale_prior = nn.Parameter(torch.zeros(self.n_scales))
        
        # Temperature for scale softmax (learnable for adaptive sharpness)
        self.temperature = nn.Parameter(torch.ones(1))
        
        # ========== Multi-Scale Patch Embedding Module ==========
        self.padding_layers = nn.ModuleList([
            ReplicationPad1d((0, st)) for st in self.strides
        ])
        
        self.patch_value_embeddings = nn.ModuleList([
            TokenEmbedding(pl, self.d_model)
            for pl in self.candidate_patch_lens
        ])
        
        # Scale-specific positional encodings
        self.scale_pos_encodings = nn.ParameterList([
            nn.Parameter(torch.randn(1, pn, self.d_model) * 0.02)
            for pn in self.patch_nums_list
        ])
        
        # Scale-specific projection (to allow scale-dependent transformations)
        self.scale_projections = nn.ModuleList([
            nn.Linear(self.d_model, self.d_model)
            for _ in range(self.n_scales)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        print(f"[FrequencyAwarePatchBlock] Initialized with:")
        print(f"  - Candidate patch lengths: {self.candidate_patch_lens}")
        print(f"  - Strides: {self.strides}")
        print(f"  - Patches per scale: {self.patch_nums_list}")
        print(f"  - Total patches: {self.total_patch_nums}")
    
    def compute_frequency_features(self, x):
        """
        Compute frequency-domain features using FFT.
        
        Args:
            x: Input tensor of shape (B, T) or (B, T, 1)
            
        Returns:
            freq_repr: Frequency representation (B, d_model)
        """
        if x.dim() == 3:
            x = x.squeeze(-1)  # (B, T)
        
        # Apply FFT (always returns float32 for complex operations)
        x_fft = torch.fft.rfft(x.float(), dim=-1)  # (B, freq_dim) complex
        
        # Extract magnitude and phase
        magnitude = torch.abs(x_fft)  # (B, freq_dim)
        phase = torch.angle(x_fft)    # (B, freq_dim)
        
        # Normalize magnitude (log scale for better gradient flow)
        magnitude = torch.log1p(magnitude)
        
        # Cast to match model dtype (handles mixed precision training)
        target_dtype = next(self.freq_encoder.parameters()).dtype
        magnitude = magnitude.to(target_dtype)
        phase = phase.to(target_dtype)
        
        # Encode magnitude and phase separately
        mag_embed = self.freq_encoder(magnitude)    # (B, d_model)
        phase_embed = self.phase_encoder(phase)     # (B, d_model)
        
        # Fuse magnitude and phase information
        freq_repr = self.freq_fusion(
            torch.cat([mag_embed, phase_embed], dim=-1)
        )  # (B, d_model)
        
        return freq_repr
    
    def compute_scale_weights(self, freq_repr):
        """
        Compute attention weights for each patch scale based on frequency content.
        
        Args:
            freq_repr: Frequency representation (B, d_model)
            
        Returns:
            scale_weights: Soft attention weights for each scale (B, n_scales)
        """
        B = freq_repr.shape[0]
        
        # Expand frequency representation for attention
        freq_repr = freq_repr.unsqueeze(1)  # (B, 1, d_model)
        
        # Expand scale queries for batch
        scale_queries = self.scale_queries.unsqueeze(0).expand(B, -1, -1)  # (B, n_scales, d_model)
        
        # Cross-attention: scale queries attend to frequency content
        scale_repr, attn_weights = self.scale_attention(
            query=scale_queries,
            key=freq_repr,
            value=freq_repr
        )  # scale_repr: (B, n_scales, d_model)
        
        # Project to scale logits
        scale_logits = self.scale_weight_proj(scale_repr).squeeze(-1)  # (B, n_scales)
        
        # Add learnable prior bias
        scale_logits = scale_logits + self.scale_prior
        
        # Apply temperature-scaled softmax
        temperature = torch.clamp(self.temperature, min=0.1, max=10.0)
        scale_weights = F.softmax(scale_logits / temperature, dim=-1)  # (B, n_scales)
        
        return scale_weights, attn_weights
    
    def create_patches(self, x, scale_idx):
        """
        Create patches for a specific scale.
        
        Args:
            x: Input tensor (B, N, T) where N is n_vars
            scale_idx: Index of the scale to use
            
        Returns:
            patches: Patch embeddings (B*N, num_patches, d_model)
            n_vars: Number of variables
        """
        n_vars = x.shape[1]
        patch_len = self.candidate_patch_lens[scale_idx]
        stride = self.strides[scale_idx]
        
        # Pad sequence
        x_padded = self.padding_layers[scale_idx](x)  # (B, N, T + stride)
        
        # Unfold to create patches
        patches = x_padded.unfold(dimension=-1, size=patch_len, step=stride)
        # patches: (B, N, num_patches, patch_len)
        
        # Reshape for embedding
        B, N, num_patches, pl = patches.shape
        patches = patches.reshape(B * N, num_patches, pl)  # (B*N, num_patches, patch_len)
        
        # Apply patch embedding
        patches = self.patch_value_embeddings[scale_idx](patches)  # (B*N, num_patches, d_model)
        
        # Add positional encoding
        patches = patches + self.scale_pos_encodings[scale_idx]
        
        # Apply scale-specific projection
        patches = self.scale_projections[scale_idx](patches)
        
        return patches, n_vars
    
    def forward(self, x):
        """
        Forward pass with learnable frequency-aware multi-scale patching.
        
        Args:
            x: Input tensor (B, N, T) after permutation, where:
               B = batch size, N = number of variables, T = sequence length
               
        Returns:
            enc_out: Multi-scale patch embeddings (B*N, total_patches, d_model)
            n_vars: Number of variables
            scale_weights: Learned scale attention weights (B*N, n_scales) - for analysis
        """
        B, N, T = x.shape
        
        # Reshape for frequency analysis: (B*N, T)
        x_flat = x.reshape(B * N, T)
        
        # ========== Frequency Analysis ==========
        freq_repr = self.compute_frequency_features(x_flat)  # (B*N, d_model)
        
        # ========== Compute Scale Weights ==========
        scale_weights, _ = self.compute_scale_weights(freq_repr)  # (B*N, n_scales)
        
        # ========== Multi-Scale Patching with Learned Weights ==========
        all_patches = []
        
        for scale_idx in range(self.n_scales):
            # Create patches for this scale
            patches, n_vars = self.create_patches(x, scale_idx)  # (B*N, num_patches_i, d_model)
            
            # Get weight for this scale
            weight = scale_weights[:, scale_idx:scale_idx+1].unsqueeze(-1)  # (B*N, 1, 1)
            
            # Weight the patches by scale importance
            weighted_patches = patches * weight
            
            all_patches.append(weighted_patches)
        
        # Concatenate all scales
        enc_out = torch.cat(all_patches, dim=1)  # (B*N, total_patches, d_model)
        enc_out = self.dropout(enc_out)
        
        return enc_out, n_vars, scale_weights
    
    def get_scale_statistics(self, scale_weights):
        """
        Utility function to analyze learned scale preferences.
        
        Args:
            scale_weights: Scale weights tensor (B, n_scales)
            
        Returns:
            dict: Statistics about scale usage
        """
        with torch.no_grad():
            mean_weights = scale_weights.mean(dim=0)
            std_weights = scale_weights.std(dim=0)
            dominant_scale = torch.argmax(mean_weights).item()
            
            return {
                'mean_weights': mean_weights.cpu().numpy(),
                'std_weights': std_weights.cpu().numpy(),
                'dominant_patch_len': self.candidate_patch_lens[dominant_scale],
                'patch_lens': self.candidate_patch_lens,
                'temperature': self.temperature.item()
            }


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        # ===== PATCHING MODE SELECTION =====
        # Options: 'frequency_aware' (learnable), 'multi_scale' (fixed), 'single' (original)
        self.patching_mode = getattr(configs, 'patching_mode', 'frequency_aware')
        
        # Get custom candidate patch lengths if provided
        self.candidate_patch_lens = getattr(configs, 'candidate_patch_lens', None)
        
        if self.patching_mode == 'frequency_aware':
            # ===== LEARNABLE FREQUENCY-AWARE PATCHING =====
            self.freq_patch_block = FrequencyAwarePatchBlock(
                configs, 
                candidate_patch_lens=self.candidate_patch_lens,
                dropout=configs.dropout
            )
            self.total_patch_nums = self.freq_patch_block.total_patch_nums
            print(f"[Frequency-Aware] Using learnable multi-scale patching")
            
        elif self.patching_mode == 'multi_scale':
            # ===== FIXED MULTI-SCALE PATCHING (Legacy) =====
            # FFT-optimized patches for weather (seq_len=96)
            self.patch_lens = [8, 16, 32]   # Short, Medium, Long
            self.strides = [4, 8, 16]       # 50% overlap
            
            self.patch_nums_list = [
                int((configs.seq_len - pl) / st + 2) 
                for pl, st in zip(self.patch_lens, self.strides)
            ]
            self.total_patch_nums = sum(self.patch_nums_list)
            print(f"[Multi-Scale] patches={self.patch_lens}, total={self.total_patch_nums}")
        else:  # 'single' mode - original single-scale patching
            self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
            self.total_patch_nums = self.patch_nums
            print(f"[Single-Scale] patch_len={self.patch_len}, stride={self.stride}, patches={self.patch_nums}")

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        # ===== PATCH EMBEDDINGS =====
        if self.patching_mode == 'frequency_aware':
            # Patch embeddings are inside FrequencyAwarePatchBlock
            pass
        elif self.patching_mode == 'multi_scale':
            self.patch_embeddings = nn.ModuleList([
                PatchEmbedding(configs.d_model, pl, st, configs.dropout)
                for pl, st in zip(self.patch_lens, self.strides)
            ])
        else:  # 'single' mode
            self.patch_embedding = PatchEmbedding(
                configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.head_nf = self.d_ff * self.total_patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        
        # ===== PATCHING =====
        if self.patching_mode == 'frequency_aware':
            # Learnable frequency-aware multi-scale patching
            enc_out, n_vars, scale_weights = self.freq_patch_block(x_enc)
            # Store scale weights for analysis (optional)
            self._last_scale_weights = scale_weights
        elif self.patching_mode == 'multi_scale':
            # Fixed multi-scale patching
            enc_outs = []
            for patch_emb in self.patch_embeddings:
                enc_out_scale, n_vars = patch_emb(x_enc)
                enc_outs.append(enc_out_scale)
            enc_out = torch.cat(enc_outs, dim=1)  # Concatenate all scales
        else:  # 'single' mode
            enc_out, n_vars = self.patch_embedding(x_enc)
        
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.total_patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags
    
    def get_scale_analysis(self):
        """
        Get analysis of learned scale weights (only for frequency_aware mode).
        
        Returns:
            dict: Statistics about scale usage, or None if not using frequency_aware mode
        """
        if self.patching_mode != 'frequency_aware':
            return None
        
        if hasattr(self, '_last_scale_weights') and self._last_scale_weights is not None:
            return self.freq_patch_block.get_scale_statistics(self._last_scale_weights)
        return None
    
    def get_learnable_patch_params(self):
        """
        Get the learnable parameters related to patch length selection.
        Useful for analysis and debugging.
        
        Returns:
            dict: Dictionary of learnable parameters
        """
        if self.patching_mode != 'frequency_aware':
            return None
        
        return {
            'scale_queries': self.freq_patch_block.scale_queries.detach().cpu(),
            'scale_prior': self.freq_patch_block.scale_prior.detach().cpu(),
            'temperature': self.freq_patch_block.temperature.detach().cpu(),
            'candidate_patch_lens': self.freq_patch_block.candidate_patch_lens,
        }
    
    def print_patch_info(self, epoch=None):
        """
        Print the current learned patch length information.
        Call this after each epoch to monitor learning progress.
        
        Args:
            epoch: Optional epoch number for logging
        """
        if self.patching_mode != 'frequency_aware':
            print("[Patch Info] Not using frequency_aware mode, skipping...")
            return
        
        with torch.no_grad():
            patch_lens = self.freq_patch_block.candidate_patch_lens
            scale_prior = self.freq_patch_block.scale_prior.detach().float().cpu().numpy()
            temperature = self.freq_patch_block.temperature.item()
            
            # Compute softmax of scale_prior to show learned preferences
            prior_weights = torch.softmax(self.freq_patch_block.scale_prior / temperature, dim=0)
            prior_weights = prior_weights.float().cpu().numpy()
            
            # Find dominant patch length based on prior
            dominant_idx = prior_weights.argmax()
            dominant_patch_len = patch_lens[dominant_idx]
            
            # Build output string
            epoch_str = f"Epoch {epoch}" if epoch is not None else "Current"
            
            print(f"\n{'='*60}")
            print(f"[Patch Length Info] {epoch_str}")
            print(f"{'='*60}")
            print(f"  Temperature: {temperature:.4f}")
            print(f"  Scale Prior (raw):    {scale_prior}")
            print(f"  Scale Prior (softmax): {prior_weights}")
            print(f"  Candidate Patch Lengths: {patch_lens}")
            print(f"  Learned Preferences:")
            for i, (pl, pw) in enumerate(zip(patch_lens, prior_weights)):
                bar = '█' * int(pw * 30)
                marker = " ← dominant" if i == dominant_idx else ""
                print(f"    patch_len={pl:3d}: {pw:.4f} |{bar}{marker}")
            print(f"  Dominant Patch Length: {dominant_patch_len}")
            
            # Also show last batch statistics if available
            if hasattr(self, '_last_scale_weights') and self._last_scale_weights is not None:
                last_weights = self._last_scale_weights.mean(dim=0).float().cpu().numpy()
                print(f"  Last Batch Avg Weights: {last_weights}")
            print(f"{'='*60}\n")
            
            return {
                'epoch': epoch,
                'temperature': temperature,
                'scale_prior': scale_prior,
                'prior_weights': prior_weights,
                'patch_lens': patch_lens,
                'dominant_patch_len': dominant_patch_len
            }


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
