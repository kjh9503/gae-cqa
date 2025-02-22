# NELL

CUDA_VISIBLE_DEVICES=1 python main.py --dataset NELL --model ComplEx --rank 1000 --max_epochs 200 --score_rel True \
--w_rel 0.1  --learning_rate 0.1 --batch_size 1000 --lmbda 0.05