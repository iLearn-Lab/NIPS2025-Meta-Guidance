export CUDA_VISIBLE_DEVICES=0

model_name=Transformer


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/HD/ \
  --data_path HD_miss.csv \
  --model_id HD_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --des 'Exp' \
  --itr 1 \
  --d_model 32 \
  --d_ff 32

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather_miss.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --d_model 32 \
  --d_ff 32

model_name=TimesNet


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/HD/ \
  --data_path HD_miss.csv \
  --model_id HD_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --des 'Exp' \
  --itr 1 \
  --d_model 32 \
  --d_ff 32

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather_miss.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --d_model 32 \
  --d_ff 32