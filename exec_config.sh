#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:tesla-smx2:1
#SBATCH --mem-per-cpu=2G
#SBATCH -o RePaViT_small_BS1024_FFN_BatchNorm_PerOperation_ShortcutGain1.0+FinetuneAfter300Epochs_lr5e-4_warmuplr1e-6+warmup20epochs_minlr1e-5_out.txt
#SBATCH -e RePaViT_small_BS1024_FFN_BatchNorm_PerOperation_ShortcutGain1.0+FinetuneAfter300Epochs_lr5e-4_warmuplr1e-6+warmup20epochs_minlr1e-5_err.txt

#export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
#export MASTER_PORT=12390
#export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)

#srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py --data-path ../data/imagenet --batch-size 256 --model normalization_free_deit_small_patch16_224_layer12 --accumulation-steps=1 --drop-path=0.05 --output_dir=output/normalization_free_deit_small_patch16_224_layer12_BS1024_Both_BatchNorm+BNFFN_PerOperation_ShortcutGain1.0+FinetuneAfter200Epochs_lr1e-3_warmuplr1e-6+warmup20epochs_minlr1e-5 --affected_layers=Both --shortcut_type=PerOperation --feature_norm=BatchNorm --lr=1e-3 --min-lr=1e-5 --warmup-lr=1e-6 --shortcut_gain=1.0 --finetune_gain=200 --opt=nadamw --num_workers=10 --weight-decay=0.05 --warmup-epochs=20 --resume=output/normalization_free_deit_small_patch16_224_layer12_BS1024_Both_BatchNorm+BNFFN_PerOperation_ShortcutGain1.0+FinetuneAfter200Epochs_lr1e-3_warmuplr1e-6+warmup20epochs_minlr1e-5/checkpoint.pth

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_PORT=12404
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)

srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py --data-path /scratch/itee/uqxxu16/data/imagenet --batch-size 256 --model normalization_free_deit_small_patch16_224_layer12 --accumulation-steps=1 --drop-path=0.05 --output_dir=output/RePaViT_small_BS1024_FFN_BatchNorm_PerOperation_ShortcutGain1.0+FinetuneAfter300Epochs_lr5e-4_warmuplr1e-6+warmup20epochs_minlr1e-5 --affected_layers=FFN --shortcut_type=PerOperation --feature_norm=BatchNorm --lr=1e-3 --min-lr=1e-5 --warmup-lr=1e-6 --shortcut_gain=1.0 --finetune_gain=300 --opt=nadamw --num_workers=20 --weight-decay=0.05 --warmup-epochs=20