#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100 
#SBATCH --time=48:00:00
#SBATCH --mem=120G
#SBATCH --job-name=dff_train_finetune
#SBATCH --output=/scratch-shared/ljc/dff/slurm_output/train_finetune.out
#SBATCH --error=/scratch-shared/ljc/dff/slurm_output/train_finetune.err

# export LD_PRELOAD=/scratch-shared/npu/conda_envs/vit_h100/lib/libiomp5.so:$LD_PRELOAD

# module spider CUDA/12.6.0

export MASTER_ADDR="localhost"
export MASTER_PORT="12345"

module load 2024
module load CUDA/12.6.0
source /home/khe/miniconda3/bin/activate /scratch-shared/khe/conda_envs/dff_env

torchrun --nproc_per_node=1 train.py --cfg-path configs/caption_coco_ft.yaml