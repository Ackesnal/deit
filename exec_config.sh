#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:tesla-smx2:2
#SBATCH --mem-per-cpu=2G
#SBATCH -o GraphPropagation_Train_DiagAttn_ThresholdGraph_Alpha0.5_Sparsity0.2_300epoch_out.txt
#SBATCH -e GraphPropagation_Train_DiagAttn_ThresholdGraph_Alpha0.5_Sparsity0.2_300epoch_err.txt

srun python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model graph_propagation_deit_small_patch16_224 --batch-size 512 --data-path ../data/imagenet/ --output_dir ./output/GraphPropagation_Train_DiagAttn_Reduction8_ThresholdGraph_Alpha0.5_Sparsity0.2_300epoch --epochs 300 --input-size 224 --selection DiagAttn --propagation ThresholdGraph --reduction_num 8 --sparsity 0.2

srun python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model graph_propagation_deit_small_patch16_224 --batch-size 512 --data-path ../data/imagenet/ --output_dir ./output/GraphPropagation_Train_DiagAttn_Reduction11_ThresholdGraph_Alpha0.5_Sparsity0.2_300epoch --epochs 300 --input-size 224 --selection DiagAttn --propagation ThresholdGraph --reduction_num 11 --sparsity 0.2

srun python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model graph_propagation_deit_small_patch16_224 --batch-size 512 --data-path ../data/imagenet/ --output_dir ./output/GraphPropagation_Train_DiagAttn_Reduction14_ThresholdGraph_Alpha0.5_Sparsity0.2_300epoch --epochs 300 --input-size 224 --selection DiagAttn --propagation ThresholdGraph --reduction_num 14 --sparsity 0.2

srun python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model graph_propagation_deit_small_patch16_224 --batch-size 512 --data-path ../data/imagenet/ --output_dir ./output/GraphPropagation_Train_DiagAttn_Reduction16_ThresholdGraph_Alpha0.5_Sparsity0.2_300epoch --epochs 300 --input-size 224 --selection DiagAttn --propagation ThresholdGraph --reduction_num 16 --sparsity 0.2

srun python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model graph_propagation_deit_small_patch16_224 --batch-size 512 --data-path ../data/imagenet/ --output_dir ./output/GraphPropagation_Train_DiagAttn_Reduction4_ThresholdGraph_Alpha0.5_Sparsity0.2_300epoch --epochs 300 --input-size 224 --selection DiagAttn --propagation ThresholdGraph --reduction_num 4 --sparsity 0.2

