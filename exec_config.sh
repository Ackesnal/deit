BATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=Ackesnal
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=5G
#SBATCH -o dense_predict_deit_tiny_out.txt
#SBATCH -e dense_predict_deit_tiny_err.txt

srun python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model deit_tiny_shuffle_patch16_224 --batch-size 512 --data-path ../data/imagenet/ --output_dir ./output/dense_predict_deit_tiny


