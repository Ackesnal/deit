#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:tesla-smx2:1
#SBATCH --mem-per-cpu=2G
#SBATCH -o normalization_free_deit_small_patch16_224_layer12_BS512_WARMUP1e-5_LR5e-4_AGCCLIP0.1_out.txt
#SBATCH -e normalization_free_deit_small_patch16_224_layer12_BS512_WARMUP1e-5_LR5e-4_AGCCLIP0.1_err.txt

srun python -m torch.distributed.launch --nproc_per_node=1 --master_port=14565 --use_env main.py --data-path ../data/imagenet --batch-size 128 --model normalization_free_deit_small_patch16_224_layer12 --accumulation-steps=4 --drop-path=0.05 --output_dir normalization_free_deit_small_patch16_224_layer12 --warmup-lr=1e-5 --lr=5e-4 --clip-grad=0.1