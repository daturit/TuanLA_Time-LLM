model_name=TimeLLM
train_epochs=10
learning_rate=0.01
llama_layers=32

master_port=00097
num_process=8
batch_size=24
d_model=16
d_ff=32

comment='TimeLLM-Weather'

accelerate launch \
  --num_processes 1 \
  run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_512_96 \
  --model $model_name \
  --data Weather \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 32 \
  --d_ff 32 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_model GPT2 \
  --llm_dim 768 \
  --llm_layers $llm_layers \
  --train_epochs $train_epochs \
  --model_comment $comment
