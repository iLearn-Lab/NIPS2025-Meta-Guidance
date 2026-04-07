export CUDA_VISIBLE_DEVICES=5

# model_name=Transformer
# model_name=TimesNet
model_name=iTransformer_my
rates=(0.1 0.25 0.4)
seeds=(1 2 3)

##############################################
for rate in "${rates[@]}"
do
  for seed in "${seeds[@]}"
  do
    python -u myrun.py \
      --task_name imputation \
      --is_training 1 \
      --root_path ./dataset/weather/ \
      --data_path weather.csv \
      --model_id weather \
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
      --batch_size 32 \
      --d_model 128 \
      --d_ff 128 \
      --des 'Exp' \
      --itr 1 \
      --top_k 3 \
      --learning_rate 0.001 \
      --mask_rate $rate \
      --seed $seed
  done
done

# ##############################################
# for rate in "${rates[@]}"
# do
#   for seed in "${seeds[@]}"
#   do
#     python -u myrun.py \
#       --task_name imputation \
#       --is_training 1 \
#       --root_path ./dataset/electricity/ \
#       --data_path electricity.csv \
#       --model_id electricity \
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
#       --batch_size 32 \
#       --d_model 128 \
#       --d_ff 128 \
#       --des 'Exp' \
#       --itr 1 \
#       --top_k 3 \
#       --learning_rate 0.001 \
#       --mask_rate $rate \
#       --seed $seed
#   done
# done

# ##############################################
# for rate in "${rates[@]}"
# do
#   for seed in "${seeds[@]}"
#   do
#     python -u myrun.py \
#       --task_name imputation \
#       --is_training 1 \
#       --root_path ./dataset/traffic/ \
#       --data_path traffic.csv \
#       --model_id traffic \
#       --model $model_name \
#       --data MyData \
#       --features M \
#       --seq_len 96 \
#       --label_len 0 \
#       --pred_len 0 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 862 \
#       --dec_in 862 \
#       --c_out 862 \
#       --batch_size 32 \
#       --d_model 128 \
#       --d_ff 128 \
#       --des 'Exp' \
#       --itr 1 \
#       --top_k 3 \
#       --learning_rate 0.001 \
#       --mask_rate $rate \
#       --seed $seed
#   done
# done

