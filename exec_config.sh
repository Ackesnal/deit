#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:tesla-smx2:2
#SBATCH --mem-per-cpu=2G
#SBATCH -o normalization_free_deit_small_patch16_224_layer12_BS1024_Both_BatchNorm_PerOperation_ShortcutGain1.0+FinetuneAfter0Epochs_lr5e-4_warmuplr1e-5_minlr1e-5_out.txt
#SBATCH -e normalization_free_deit_small_patch16_224_layer12_BS1024_Both_BatchNorm_PerOperation_ShortcutGain1.0+FinetuneAfter0Epochs_lr5e-4_warmuplr1e-5_minlr1e-5_err.txt

# 获取此任务的Slurm节点名称
node_name=$(hostname)

# 指定只有在特定节点上执行输出操作的节点名
# 例如，仅第一个节点（通常是master节点）
master_node=$(srun --ntasks=1 hostname | head -n 1)

# 检查当前节点是否是指定的master节点
if [ "$node_name" = "$master_node" ]; then
    echo "Running on the master node: $node_name"
    # 运行你的命令或脚本，并产生输出
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=14577 --use_env main.py --data-path ../data/imagenet --batch-size 256 --model normalization_free_deit_small_patch16_224_layer12 --accumulation-steps=1 --drop-path=0.05 --output_dir=output/normalization_free_deit_small_patch16_224_layer12_BS1024_Both_BatchNorm_PerOperation_ShortcutGain1.0+FinetuneAfter0Epochs_lr5e-4_warmuplr1e-5_minlr1e-5 --affected_layers=Both --shortcut_type=PerOperation --feature_norm=BatchNorm --lr=5e-4 --min-lr=1e-5 --warmup-lr=1e-5 --shortcut_gain=1.0 --finetune_gain=0 --opt=nadamw --num_workers=10
else
    echo "Running on a worker node: $node_name"
    # 在其他节点上运行，但不产生主要输出
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=14577 --use_env main.py --data-path ../data/imagenet --batch-size 256 --model normalization_free_deit_small_patch16_224_layer12 --accumulation-steps=1 --drop-path=0.05 --output_dir=output/normalization_free_deit_small_patch16_224_layer12_BS1024_Both_BatchNorm_PerOperation_ShortcutGain1.0+FinetuneAfter0Epochs_lr5e-4_warmuplr1e-5_minlr1e-5 --affected_layers=Both --shortcut_type=PerOperation --feature_norm=BatchNorm --lr=5e-4 --min-lr=1e-5 --warmup-lr=1e-5 --shortcut_gain=1.0 --finetune_gain=0 --opt=nadamw --num_workers=10 > /dev/null 2>&1
fi

