export CUDA_VISIBLE_DEVICES=3

rates=(0.1)
seeds=(2 3 4)

# for rate in "${rates[@]}"
# do
#   for seed in "${seeds[@]}"
#   do
#     echo $rate $seed
#     python -u myrun.py \
#       --task_name imputation \
#       --is_training 1 \
#       --root_path ./dataset/weather/ \
#       --data_path weather.csv \
#       --model_id weather_ \
#       --model TimesNet \
#       --data MyData \
#       --features M \
#       --seq_len 96 \
#       --label_len 0 \
#       --pred_len 0 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 21 \
#       --dec_in 21 \
#       --c_out 21 \
#       --batch_size 16 \
#       --d_model 64 \
#       --d_ff 64 \
#       --des 'Exp' \
#       --itr 1 \
#       --top_k 3 \
#       --learning_rate 0.001 \
#       --mask_rate $rate \
#       --seed $seed
#   done
# done

for rate in "${rates[@]}"
do
  for seed in "${seeds[@]}"
  do
    echo $rate $seed
    python -u myrun.py \
      --task_name imputation \
      --is_training 1 \
      --root_path ./dataset/weather/ \
      --data_path weather.csv \
      --model_id weather_ \
      --model myTimesNet \
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
      --seed $seed
  done
done

