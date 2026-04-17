"""
 Automated weight downloader for DFF.
 Downloads LLMs, Visual Backbones, and Fine-tuned checkpoints 
 and organizes them into the 'model_parameter' directory.
"""

import os
from huggingface_hub import snapshot_download, hf_hub_download
import requests
from tqdm import tqdm

def download_file(url, save_path):
    """
    Downloads a file from a direct URL with a progress bar.
    """
    if os.path.exists(save_path):
        print(f"--- File already exists: {save_path} ---")
        return

    print(f"--- Downloading: {url} ---")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as f, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def main():
    # Define root directory for models
    base_dir = "model_parameter"
    os.makedirs(base_dir, exist_ok=True)

    # 1. Download Base Language Models (Hugging Face Snapshots)
    print("\n[Step 1/3] Downloading Base Language Models...")
    
    # Flan-T5 XL
    snapshot_download(
        repo_id="google/flan-t5-xl",
        local_dir=os.path.join(base_dir, "flant5xl"),
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"] # Optional: save space by ignoring non-pytorch weights
    )

    # BERT Base Uncased
    snapshot_download(
        repo_id="google-bert/bert-base-uncased",
        local_dir=os.path.join(base_dir, "bert-base-uncased")
    )

    # 2. Download Pre-trained Vision/Multimodal Weights (Direct URLs)
    print("\n[Step 2/3] Downloading Pre-trained Multimodal Weights...")
    
    weights_urls = {
        "eva_vit_g.pth": "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth",
        "instruct_blip_flanxl_trimmed.pth": "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_flanxl_trimmed.pth"
    }

    for name, url in weights_urls.items():
        save_path = os.path.join(base_dir, name)
        download_file(url, save_path)

    # 3. Download Our Fine-tuned Models (Hugging Face Repos)
    print("\n[Step 3/3] Downloading DFF Fine-tuned Checkpoints...")
    
    # Face-ViT Classifier
    # Note: Adjust the folder name to match your multi_label_classifier_path in config
    classifier_dir = os.path.join(base_dir, "ViT-H_14-add_cnn_branch_3_5-add_max_pool-21_class-weight_1_02-bce_focal_dice_jaccard_loss")
    os.makedirs(classifier_dir, exist_ok=True)
    
    hf_hub_download(
        repo_id="LianJC/Face-ViT-MultiLabel",
        filename="Fakeface-ViT-H_14-add_cnn_branch_3_5-add_max_pool-21_class-weight_1_02-bce_focal_dice_jaccard_loss_checkpoint.bin",
        local_dir=classifier_dir
    )

    # Final DFF Model
    hf_hub_download(
        repo_id="LianJC/DFF-InstructBLIP-Detection",
        filename="2111_checkpoint_best.pth",
        local_dir=base_dir
    )

    print("\n✅ All weights downloaded and organized successfully!")

if __name__ == "__main__":
    main()