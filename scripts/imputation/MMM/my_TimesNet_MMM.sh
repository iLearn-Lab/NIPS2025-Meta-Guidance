export CUDA_VISIBLE_DEVICES=0

model_name=myTimesNet

rate=0.1
seed=4
echo $rate $seed
python -u myrun.py \
      --task_name imputation \
      --is_training 1 \
      --root_path ./dataset/MMM/ \
      --data_path MMM.csv \
      --model_id my_MMM \
      --model $model_name \
      --data MyData \
      --features M \
      --seq_len 96 \
      --label_len 0 \
      --pred_len 0 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 5 \
      --dec_in 5 \
      --c_out 5 \
      --batch_size 16 \
      --d_model 16 \
      --d_ff 32 \
      --des 'Exp' \
      --itr 1 \
      --top_k 3 \
      --learning_rate 0.001 \
      --mask_rate $rate \
      --seed $seed
echo '上述结果是' $rate $seed

