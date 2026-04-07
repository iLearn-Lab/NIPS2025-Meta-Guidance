export CUDA_VISIBLE_DEVICES=3

# model_name=myTransformer
model_name=PatchTST_my
# ks=(1 4 8 12 16 20)
# rs=(1 2 3 4 5)
rates=(0.1)
seeds=(1)

##############################################
for rate in "${rates[@]}"
do
  for seed in "${seeds[@]}"
  do
    python -u myrun.py \
      --task_name imputation \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_norev \
      --model $model_name \
      --data MyData \
      --features M \
      --seq_len 96 \
      --label_len 0 \
      --pred_len 0 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --batch_size 16 \
      --d_model 16 \
      --d_ff 32 \
      --des 'Exp' \
      --itr 1 \
      --top_k 3 \
      --learning_rate 0.001 \
      --mask_rate $rate \
      --seed $seed
  done
done

##############################################
for rate in "${rates[@]}"
do
  for seed in "${seeds[@]}"
  do
    python -u myrun.py \
      --task_name imputation \
      --is_training 1 \
      --root_path ./dataset/TCPC/ \
      --data_path TCPC.csv \
      --model_id TCPC_norev \
      --model $model_name \
      --data MyData \
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
      --mask_rate $rate \
      --seed $seed
  done
done