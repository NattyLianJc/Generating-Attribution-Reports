#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100 
#SBATCH --time=48:00:00
#SBATCH --mem=120G
#SBATCH --job-name=test
#SBATCH --output=/scratch-shared/ljc-1/dff/slurm_output/pro.out
#SBATCH --error=/scratch-shared/ljc-1/dff/slurm_output/pro.err

# export LD_PRELOAD=/scratch-shared/npu/conda_envs/vit_h100/lib/libiomp5.so:$LD_PRELOAD

# module spider CUDA/12.6.0

export MASTER_ADDR="localhost"
export MASTER_PORT="12345"

module load 2024
module load CUDA/12.6.0
source /home/npu/miniconda3/bin/activate /scratch-shared/npu/conda_envs/dff

# python3 train.py --name Fakeface-ViT-H_14-add_cnn_branch_3_5-add_max_pool-21_class-weight_1_02-bce_focal_dice_jaccard_loss_new --dataset custom --model_type ViT-H_14 --pretrained_dir checkpoint/pytorch_model.bin --output_dir output/ViT-H_14-add_cnn_branch_3_5-add_max_pool-21_class-weight_1_02-bce_focal_dice_jaccard_loss_new --fp16 --fp16_opt_level O2

python process_dataset.py \
    --input_json_name eval-mutilabel.json \
    --dir_downscale eval_downscale_05x \
    --dir_blur eval_blur_11 \
    --dir_both eval_both_05x_11 \
    --json_downscale eval-mutilabel_downscal_05x.json \
    --json_blur eval-mutilabel_blur_11.json \
    --json_both eval-mutilabel_both_05x_11.json \
    --scale 0.5 \
    --kernel_size 11 \