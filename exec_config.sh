#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=Ackesnal
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=10G
#SBATCH -o shuffle_deit_tiny_out.txt
#SBATCH -e shuffle_deit_tiny_err.txt

srun python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_tiny_shuffle_patch16_224 --batch-size 512 --data-path ../../uqxxu16/data/imagenet/ --output_dir ./output/shuffle_deit_tiny