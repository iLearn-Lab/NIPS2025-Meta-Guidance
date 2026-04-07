export CUDA_VISIBLE_DEVICES=0

model_name=myTransformer
# model_name=myTimesNet
ks=(1 4 8 12 16 20)
# rs=(1 2 3 4 5)
rs=(3)
# rates=(0.1 0.25 0.4)
# seeds=(1)
rate=0.1
seed=1

# ##############################################
# for k in "${ks[@]}"
# do
#   for r in "${rs[@]}"
#   do
#     python -u myrun.py \
#       --task_name imputation \
#       --is_training 1 \
#       --root_path ./dataset/HD/ \
#       --data_path HD.csv \
#       --model_id HD_norev \
#       --model $model_name \
#       --data MyData \
#       --features M \
#       --seq_len 96 \
#       --label_len 0 \
#       --pred_len 0 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 5 \
#       --dec_in 5 \
#       --c_out 5 \
#       --batch_size 16 \
#       --d_model 16 \
#       --d_ff 32 \
#       --des 'Exp' \
#       --itr 1 \
#       --top_k 3 \
#       --learning_rate 0.001 \
#       --mask_rate $rate \
#       --seed $seed \
      # --k $k \
      # --r $r
#   done
# done

##############################################
for k in "${ks[@]}"
do
  for r in "${rs[@]}"
  do
    python -u myrun.py \
      --task_name imputation \
      --is_training 1 \
      --root_path ./dataset/weather/ \
      --data_path weather.csv \
      --model_id Weather_norev \
      --model $model_name \
      --data MyData \
      --features M \
      --seq_len 96 \
      --label_len 0 \
      --pred_len 0 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --batch_size 16 \
      --d_model 64 \
      --d_ff 64 \
      --des 'Exp' \
      --itr 1 \
      --top_k 3 \
      --learning_rate 0.001 \
      --mask_rate $rate \
      --seed $seed \
      --k $k \
      --r $r
  done
done

# ##############################################
# for rate in "${rates_tmp[@]}"
# do
#   for r in "${rs[@]}"
#   do
#     python -u myrun.py \
#       --task_name imputation \
#       --is_training 1 \
#       --root_path ./dataset/electricity/ \
#       --data_path electricity.csv \
#       --model_id electricity_w_rev \
#       --model $model_name \
#       --data MyData \
#       --features M \
#       --seq_len 96 \
#       --label_len 0 \
#       --pred_len 0 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 321 \
#       --dec_in 321 \
#       --c_out 321 \
#       --batch_size 16 \
#       --d_model 128 \
#       --d_ff 128 \
#       --des 'Exp' \
#       --itr 1 \
#       --top_k 3 \
#       --learning_rate 0.001 \
#       --mask_rate $rate \
#       --seed $seed \
      # --k $k \
      # --r $r
#   done
# done

# ##############################################
# for k in "${ks[@]}"
# do
#   for r in "${rs[@]}"
#   do
#     python -u myrun.py \
#       --task_name imputation \
#       --is_training 1 \
#       --root_path ./dataset/TCPC/ \
#       --data_path TCPC.csv \
#       --model_id TCPC_norev \
#       --model $model_name \
#       --data MyData \
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
#       --mask_rate $rate \
#       --seed $seed \
      # --k $k \
      # --r $r
#   done
# done


############################################
for k in "${ks[@]}"
do
  for r in "${rs[@]}"
  do
    python -u myrun.py \
      --task_name imputation \
      --is_training 1 \
      --root_path ./dataset/traffic/ \
      --data_path traffic.csv \
      --model_id traffic_w_rev \
      --model $model_name \
      --data MyData \
      --features M \
      --seq_len 96 \
      --label_len 0 \
      --pred_len 0 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --batch_size 16 \
      --d_model 512 \
      --d_ff 512 \
      --des 'Exp' \
      --itr 1 \
      --top_k 3 \
      --learning_rate 0.001 \
      --mask_rate $rate \
      --seed $seed \
      --k $k \
      --r $r
  done
done

# ##############################################
# for r in "${rs[@]}"
# do
#   for k in "${ks[@]}"
#   do
#     python -u myrun.py \
#       --task_name imputation \
#       --is_training 1 \
#       --root_path ./dataset/ETT-small/ \
#       --data_path ETTh1.csv \
#       --model_id ETTh1_norev \
#       --model $model_name \
#       --data MyData \
#       --features M \
#       --seq_len 96 \
#       --label_len 0 \
#       --pred_len 0 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 7 \
#       --dec_in 7 \
#       --c_out 7 \
#       --batch_size 16 \
#       --d_model 16 \
#       --d_ff 32 \
#       --des 'Exp' \
#       --itr 1 \
#       --top_k 3 \
#       --learning_rate 0.001 \
#       --mask_rate $rate \
#       --seed 2 \
#       --k 3 \
#       --r $r
#   done
# done

# ##############################################
# for k in "${ks[@]}"
# do
#   for k in "${ks[@]}"
#   do
#     python -u myrun.py \
#       --task_name imputation \
#       --is_training 1 \
#       --root_path ./dataset/ETT-small/ \
#       --data_path ETTh1.csv \
#       --model_id ETTh1_norev \
#       --model $model_name \
#       --data MyData \
#       --features M \
#       --seq_len 96 \
#       --label_len 0 \
#       --pred_len 0 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 7 \
#       --dec_in 7 \
#       --c_out 7 \
#       --batch_size 16 \
#       --d_model 16 \
#       --d_ff 32 \
#       --des 'Exp' \
#       --itr 1 \
#       --top_k 3 \
#       --learning_rate 0.001 \
#       --mask_rate $rate \
#       --seed 2 \
#       --k $k \
#       --r 3
#   done
# done

# ##############################################
# for k in "${ks[@]}"
# do
#   for r in "${rs[@]}"
#   do
#     python -u myrun.py \
#       --task_name imputation \
#       --is_training 1 \
#       --root_path ./dataset/ETT-small/ \
#       --data_path ETTh2.csv \
#       --model_id ETTh2_norev \
#       --model $model_name \
#       --data MyData \
#       --features M \
#       --seq_len 96 \
#       --label_len 0 \
#       --pred_len 0 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 7 \
#       --dec_in 7 \
#       --c_out 7 \
#       --batch_size 16 \
#       --d_model 16 \
#       --d_ff 32 \
#       --des 'Exp' \
#       --itr 1 \
#       --top_k 3 \
#       --learning_rate 0.001 \
#       --mask_rate $rate \
#       --seed $seed \
      # --k $k \
      # --r $r
#   done
# done

# ##############################################
# for k in "${ks[@]}"
# do
#   for r in "${rs[@]}"
#   do
#     python -u myrun.py \
#       --task_name imputation \
#       --is_training 1 \
#       --root_path ./dataset/ETT-small/ \
#       --data_path ETTm1.csv \
#       --model_id ETTm1_norev \
#       --model $model_name \
#       --data MyData \
#       --features M \
#       --seq_len 96 \
#       --label_len 0 \
#       --pred_len 0 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 7 \
#       --dec_in 7 \
#       --c_out 7 \
#       --batch_size 16 \
#       --d_model 16 \
#       --d_ff 32 \
#       --des 'Exp' \
#       --itr 1 \
#       --top_k 3 \
#       --learning_rate 0.001 \
#       --mask_rate $rate \
#       --seed $seed \
      # --k $k \
      # --r $r
#   done
# done

# ##############################################
# for k in "${ks[@]}"
# do
#   for r in "${rs[@]}"
#   do
#     python -u myrun.py \
#       --task_name imputation \
#       --is_training 1 \
#       --root_path ./dataset/ETT-small/ \
#       --data_path ETTm2.csv \
#       --model_id ETTm2_norev \
#       --model $model_name \
#       --data MyData \
#       --features M \
#       --seq_len 96 \
#       --label_len 0 \
#       --pred_len 0 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 7 \
#       --dec_in 7 \
#       --c_out 7 \
#       --batch_size 16 \
#       --d_model 16 \
#       --d_ff 32 \
#       --des 'Exp' \
#       --itr 1 \
#       --top_k 3 \
#       --learning_rate 0.001 \
#       --mask_rate $rate \
#       --seed $seed \
      # --k $k \
      # --r $r
#   done
# done