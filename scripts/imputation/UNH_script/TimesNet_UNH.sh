export CUDA_VISIBLE_DEVICES=6

model_name=TimesNet

rate=0.1
seed=2
# for rate in {0.4}
# do
  # for seed in {2,16}
  # do
    echo $rate $seed
    python -u run.py \
      --task_name imputation \
      --is_training 1 \
      --root_path ./dataset/UNH/ \
      --data_path UNH.csv \
      --model_id UNH_0.1_2 \
      --mask_rate $rate \
      --model $model_name \
      --data custom \
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
      --seed $seed
    echo '上述结果是' $rate $seed
  # done
# done
