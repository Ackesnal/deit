#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=Ackesnal
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:tesla-smx2:2
#SBATCH --mem-per-cpu=10G
#SBATCH -o modified3_new3_shuffle_deit_tiny_out.txt
#SBATCH -e modified3_new3_shuffle_deit_tiny_err.txt


"""
for i in {1..36}; do
    srun python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_tiny_shuffle_patch16_224 --batch-size 2048 --data-path ../../uqxxu16/data/imagenet/ --output_dir ./output/modified3_new3_shuffle_deit_tiny --resume ./output/modified3_new3_shuffle_deit_tiny/checkpoint.pth
    wait
done
"""
srun python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model deit_tiny_shuffle_patch16_224 --batch-size 2048 --data-path ../../uqxxu16/data/imagenet/ --output_dir ./output/modified3_new4_shuffle_deit_tiny --resume ./output/modified3_new_shuffle_deit_tiny/checkpoint.pth
