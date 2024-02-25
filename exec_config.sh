#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH -o normalization_free_deit_small_patch16_224_layer12_BS1024_WARMUP1e-5_LR5e-3_AGCCLIP0.1_out.txt
#SBATCH -e normalization_free_deit_small_patch16_224_layer12_BS1024_WARMUP1e-5_LR5e-3_AGCCLIP0.1_err.txt

srun python -m torch.distributed.launch --nproc_per_node=1 --master_port=14575 --use_env main.py --data-path ../data/imagenet --batch-size 128 --model normalization_free_deit_small_patch16_224_layer12 --accumulation-steps=8 --drop-path=0.05 --output_dir normalization_free_deit_small_patch16_224_layer12 --warmup-lr=1e-5 --lr=5e-3 --clip-grad=0.1