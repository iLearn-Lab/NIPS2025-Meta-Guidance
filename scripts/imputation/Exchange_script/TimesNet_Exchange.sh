export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

# for rate in {0.1,0.4,0.7}
# do
#   for seed in {2,16,71}
#   do
#     echo $rate $seed
#     python -u run.py \
#       --task_name imputation \
#       --is_training 0 \
#       --root_path ./dataset/Exchange/ \
#       --data_path exchange_rate.csv \
#       --model_id Exchange_mask_0.1 \
#       --mask_rate $rate \
#       --model $model_name \
#       --data custom \
#       --features M \
#       --seq_len 96 \
#       --label_len 0 \
#       --pred_len 0 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 8 \
#       --dec_in 8 \
#       --c_out 8 \
#       --batch_size 16 \
#       --d_model 16 \
#       --d_ff 32 \
#       --des 'Exp' \
#       --itr 1 \
#       --top_k 3 \
#       --learning_rate 0.001 \
#       --seed $seed
#     echo '上述结果是' $rate $seed
#   done
# done

python -u run.py \
      --task_name imputation \
      --is_training 0 \
      --root_path ./dataset/Exchange/ \
      --data_path exchange_rate.csv \
      --model_id Exchange_mask_0.1 \
      --mask_rate 0.7 \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --label_len 0 \
      --pred_len 0 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --batch_size 16 \
      --d_model 16 \
      --d_ff 32 \
      --des 'Exp' \
      --itr 1 \
      --top_k 3 \
      --learning_rate 0.001 \
      --seed 2


# python -u run.py \
#   --task_name imputation \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_mask_0.25 \
#   --mask_rate 0.25 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 0 \
#   --pred_len 0 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --batch_size 16 \
#   --d_model 16 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 3 \
#   --learning_rate 0.001

# python -u run.py \
#   --task_name imputation \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_mask_0.375 \
#   --mask_rate 0.375 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 0 \
#   --pred_len 0 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --batch_size 16 \
#   --d_model 16 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 3 \
#   --learning_rate 0.001

# python -u run.py \
#   --task_name imputation \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_mask_0.5 \
#   --mask_rate 0.5 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 0 \
#   --pred_len 0 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --batch_size 16 \
#   --d_model 16 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 3 \
#   --learning_rate 0.001
