#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:tesla-smx2:1
#SBATCH --mem-per-cpu=2G
#SBATCH -o train_out.txt
#SBATCH -e train_err.txt

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_PORT=12580
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export BATCH_SIZE=2048 / $WORLD_SIZE

# srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py --data-path /scratch/itee/uqxxu16/data/imagenet --batch-size 256 --model normalization_free_deit_small_patch16_224_layer12 --accumulation-steps=1 --drop-path=0.05 --output_dir=output/RePaViT_small_BS1024_Both_Fixup_BatchNorm_PerOperation_ShortcutGain0.2+FinetuneAfter300Epochs_lr5e-4_warmuplr1e-6+warmup20epochs_minlr1e-6 --affected_layers=Both --shortcut_type=PerOperation --feature_norm=BatchNorm --lr=5e-4 --min-lr=1e-6 --warmup-lr=1e-6 --shortcut_gain=0.2 --finetune_gain=300 --opt=nadamw --num_workers=15 --weight-decay=0.05 --warmup-epochs=20

srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py --data-path /scratch/itee/uqxxu16/data/imagenet --batch-size=$BATCH_SIZE --model=RePaViT_small_patch16_224_layer12 --output_dir=output/debug --feature_norm=BatchNorm --shortcut_gain=0.2 --lr=5e-4 --min-lr=1e-6 --warmup-lr=1e-6 --opt=nadamw --num_workers=15 --weight-decay=0.05 --warmup-epochs=20