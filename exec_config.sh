#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=Ackesnal
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:tesla-smx2:4
#SBATCH --mem-per-cpu=10G
#SBATCH -o normed_shuffle_deit_tiny_another_out.txt
#SBATCH -e normed_shuffle_deit_tiny_another_err.txt

srun python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_tiny_shuffle_patch16_224 --batch-size 2048 --data-path ../../uqxxu16/data/imagenet/ --output_dir ./output/normed_shuffle_deit_tiny_another --resume ./output/normed_shuffle_deit_tiny_another/checkpoint.pth --min-lr 1e-8 --drop-path 0.01 --epochs 400

