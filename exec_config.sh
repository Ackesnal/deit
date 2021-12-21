#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=Ackesnal
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:tesla-smx2:2
#SBATCH --mem-per-cpu=10G
#SBATCH -o modified_new6_shuffle_deit_tiny_out.txt
#SBATCH -e modified_new6_shuffle_deit_tiny_err.txt

srun python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model deit_tiny_shuffle_patch16_224 --batch-size 1024 --data-path ../../uqxxu16/data/imagenet/ --output_dir ./output/modified3_new6_shuffle_deit_tiny --resume ./output/modified3_new_shuffle_deit_tiny/checkpoint.pth --min-lr 2e-5 --drop-path 0.01

