CUDA_VISIBLE_DEVICES=0 python pretrain_lmpnn_filled.py --cuda --do_train --data_path data/FB15k-237-betae \
--kge_ckpt_path ssl-relation-prediction/src/ckpts/FB15k-237/ComplEx-2024.03.15-16_25_39/best_valid.model \
-b 1024 -n 512 -de 2000 -dr 2000 -lr 0.0002 --label_smoothing 0.6 --cpu_num 5 --geo complex --num_hidden_layers 6 \
--num_attention_heads 16 --hidden_size 768 --intermediate_size 768 --token_embeddings 0 --hidden_dropout_prob 0.1 \
--warm_up_steps 20000 --max_steps 240000 --valid_steps 50000000 \
--recon all \
--spc_tok_mask True \
--note pretrain \
--cqd_ckpt_path pretrain/cqd/FB15k-237-model-rank-1000-epoch-100-1602508358.pt \
--mp_rel True \
--pred tgt_pred --enc_mask random --dec_mask zero #여기서 token 은 random mask 의미함

#--kge_ckpt_path ssl-relation-prediction/src/ckpts/FB15k-237/ComplEx-2024.03.15-16_25_39/best_valvscode-remote://ssh-remote%2B229/home/kjh9503/q2t_downstream/pre2-3_down1.sh