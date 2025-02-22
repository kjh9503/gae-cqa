CUDA_VISIBLE_DEVICES=0 python pretrain_lmpnn_filled.py --cuda --do_train --data_path data/NELL-betae \
--kge_ckpt_path ssl-relation-prediction/src/ckpts/NELL/ComplEx-2024.03.21-15_37_26/best_valid.model \
-b 1024 -n 512 -de 2000 -dr 2000 -lr 0.0005 --label_smoothing 0.6 --cpu_num 5 --geo complex --num_hidden_layers 6 \
--num_attention_heads 12 --hidden_size 768 --intermediate_size 768 --token_embeddings 0 --hidden_dropout_prob 0.1 \
--warm_up_steps 20000 --max_steps 240000 --valid_steps 50000000 \
--recon all \
--note pretrain \
--cqd_ckpt_path pretrain/cqd/NELL-model-rank-1000-epoch-100-1602499096.pt \
--pred tgt_pred --enc_mask random --dec_mask zero