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

# python3 train.py --name Fakeface-ViT-H_14-add_cnn_branch_3_5-add_max_pool-21_class-weight_1_02-bce_focal_dice_jaccard_loss_new --dataset custom --model_type ViT-H_14 --pretrained_dir checkpoint/pytorch_model.bin --output_dir output/ViT-H_14-add_cnn_branch_3_5-add_max_pool-21_class-weight_1_02-bce_focal_dice_jaccard_loss_new --fp16 --fp16_opt_level O2

# python mytrain.py --cfg-path ./caption_coco_ft.yaml
# python test.py \
#     --json_file_path dataset/test-mutilabel.json \
#     --output_text_file test.json \
#     --image_base_dir dataset
# python evaluate_metrics.py \
#     --pred_json test.json \
#     --gt_json dataset/test-mutilabel.json \
#     --pred_mask_base test_mask \
#     --gt_mask_base dataset \
#     --output_result_json metrics_results/test_results.json

# python evaluate_metrics.py \
#     --pred_json downscale_captions.json \
#     --gt_json dataset/eval-mutilabel_downscale.json \
#     --pred_mask_base downscale_mask \
#     --gt_mask_base dataset \
#     --output_result_json metrics_results/downscale_results.json
# python evaluate_metrics.py \
#     --pred_json blur_captions.json \
#     --gt_json dataset/eval-mutilabel_blur.json \
#     --pred_mask_base blur_mask \
#     --gt_mask_base dataset \
#     --output_result_json metrics_results/blur_results.json
# python evaluate_metrics.py \
#     --pred_json both_captions.json \
#     --gt_json dataset/eval-mutilabel_both.json \
#     --pred_mask_base both_mask \
#     --gt_mask_base dataset \
#     --output_result_json metrics_results/both_results.json
# python evaluate_metrics.py \
#     --pred_json downscale_captions_05x.json \
#     --gt_json dataset/eval-mutilabel_downscal_05x.json \
#     --pred_mask_base downscale_05x_mask \
#     --gt_mask_base dataset \
#     --output_result_json metrics_results/downscale_05x_results.json
# python evaluate_metrics.py \
#     --pred_json blur_11_captions.json \
#     --gt_json dataset/eval-mutilabel_blur_11.json \
#     --pred_mask_base blur_11_mask \
#     --gt_mask_base dataset \
#     --output_result_json metrics_results/blur_11_results.json
# python evaluate_metrics.py \
#     --pred_json both_05x_11_captions.json \
#     --gt_json dataset/eval-mutilabel_both_05x_11.json \
#     --pred_mask_base both_05x_11_mask \
#     --gt_mask_base dataset \
#     --output_result_json metrics_results/both_results_05x_11.json

# python mytrain.py --cfg-path ./caption_distill_high_lr_stage2_finetune.yaml
torchrun --nproc_per_node=1 train.py --cfg-path configs/caption_coco_ft.yaml