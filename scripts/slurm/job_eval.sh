#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100 
#SBATCH --time=48:00:00
#SBATCH --mem=120G
#SBATCH --job-name=infer
#SBATCH --output=/scratch-shared/ljc/dff/slurm_output/infer.out
#SBATCH --error=/scratch-shared/ljc/dff/slurm_output/infer.err

# export LD_PRELOAD=/scratch-shared/npu/conda_envs/vit_h100/lib/libiomp5.so:$LD_PRELOAD

# module spider CUDA/12.6.0

export MASTER_ADDR="localhost"
export MASTER_PORT="12345"

module load 2024
module load CUDA/12.6.0
source /home/khe/miniconda3/bin/activate /scratch-shared/khe/conda_envs/dff_env

python scripts/evaluation/evaluate.py \
    --checkpoint model_parameter/2111_checkpoint_best.pth \
    --json_file dataset/test-mutilabel.json \
    --output_file results/test_results.jsonl \
    --output_mask_dir results/test_masks \
    --batch_size 16 \
    --num_workers 8