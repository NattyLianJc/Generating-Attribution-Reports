#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100 
#SBATCH --time=48:00:00
#SBATCH --mem=120G
#SBATCH --job-name=eval
#SBATCH --output=/scratch-shared/ljc/dff/slurm_output/eval.out
#SBATCH --error=/scratch-shared/ljc/dff/slurm_output/eval.err

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
#     --json_file_path dataset/eval-mutilabel.json \
#     --output_text_file eval_captions.json \
#     --output_mask_dir eval_mask \
#     --image_base_dir dataset

# python evaluate_metrics.py \
#     --pred_json eval_captions.json \
#     --gt_json dataset/eval-mutilabel.json \
#     --pred_mask_base eval_mask \
#     --gt_mask_base dataset \
#     --output_result_json metrics_results/eval_results.json

# python clean_and_upload.py

pip install -r requirements.txt
# python app.py