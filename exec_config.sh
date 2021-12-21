#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=Ackesnal
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:tesla-smx2:4
#SBATCH --mem-per-cpu=10G
#SBATCH -o modified_new5_shuffle_deit_tiny_out.txt
#SBATCH -e modified_new5_shuffle_deit_tiny_err.txt

for i in {1..100}; do
    srun python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_tiny_shuffle_patch16_224 --batch-size 256 --data-path ../../uqxxu16/data/imagenet/ --output_dir ./output/modified3_new5_shuffle_deit_tiny --resume ./output/modified3_new4_shuffle_deit_tiny/checkpoint.pth
    wait
done
