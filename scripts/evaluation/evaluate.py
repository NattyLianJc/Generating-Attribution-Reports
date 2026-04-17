"""
 Evaluation script for batched inference (Captions & Masks).
 Optimized for high-performance GPUs (e.g., H100/A100).
 Now supports streaming output to JSONL for real-time validation.
"""

import json
import os
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import sys

# Ensure the script can find the 'lavis' module from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lavis.models import load_model_and_preprocess

class DeepFakeEvalDataset(Dataset):
    """
    Dataset class for efficient image loading and preprocessing.
    """
    def __init__(self, data, image_base_dir, vis_processor):
        self.data = data
        self.image_base_dir = image_base_dir
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        image_id = record.get("image_id", "unknown")
        # Handle different key names from different JSON versions
        relative_path = record.get("img_path") or record.get("relative_path") or ""
        img_path = os.path.join(self.image_base_dir, relative_path)

        try:
            raw_image = Image.open(img_path).convert("RGB")
            image_tensor = self.vis_processor["eval"](raw_image)
        except Exception as e:
            # Print error but return a dummy tensor to keep the batch consistent
            print(f"Error loading image {img_path}: {e}")
            image_tensor = torch.zeros(3, 224, 224)

        return {
            "image": image_tensor,
            "image_id": image_id,
            "relative_path": relative_path
        }

def save_mask(mask_tensor, output_dir, relative_path, threshold=0.5):
    """
    Saves the predicted mask tensor as a binary PNG image.
    """
    # mask_tensor shape: [1, 224, 224] or [224, 224]
    mask_np = mask_tensor.cpu().numpy()
    mask_single = np.squeeze(mask_np)
    
    # Binary thresholding
    mask_binary = (mask_single > threshold).astype(np.uint8) * 255
    binary_image = Image.fromarray(mask_binary, mode='L')
    
    # Ensure subdirectory exists (handling image paths with slashes)
    output_path = os.path.join(output_dir, relative_path.replace('.jpg', '.png').replace('.jpeg', '.png'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    binary_image.save(output_path)

def main():
    parser = argparse.ArgumentParser(description="High-performance evaluation for BLIP-2.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained .pth file.")
    parser.add_argument("--json_file", type=str, default="dataset/test-mutilabel.json")
    # Changed default extension to .jsonl
    parser.add_argument("--output_file", type=str, default="results/test_captions.jsonl") 
    parser.add_argument("--output_mask_dir", type=str, default="results/test_masks")
    parser.add_argument("--image_dir", type=str, default="dataset")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing model on {device}...")

    # Load model structure and visual processors
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5_instruct", model_type="flant5xl", is_eval=True, device=device
    )

    print(f"📥 Loading weights from: {args.checkpoint}")
    model.load_checkpoint(args.checkpoint)
    model.eval()

    # Load JSON data
    with open(args.json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Initialize DataLoader for asynchronous CPU-to-GPU pipe
    dataset = DeepFakeEvalDataset(data, args.image_dir, vis_processors)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        shuffle=False, 
        pin_memory=True
    )

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    print(f"⚡ Processing {len(data)} samples... Streaming output to {args.output_file}")

    # Open file in write mode before the loop
    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for batch in tqdm(dataloader, desc="Inference"):
            images = batch["image"].to(device, non_blocking=True)
            image_ids = batch["image_id"]
            rel_paths = batch["relative_path"]

            with torch.no_grad():
                # Crucial: return_seg=True enables mask generation
                texts, masks = model.generate(
                    {"image": images},
                    use_nucleus_sampling=False,
                    num_beams=5,
                    max_length=100,
                    min_length=5,
                    return_seg=True
                )

            # Iterate through batch results and write to JSONL immediately
            for i in range(len(image_ids)):
                save_mask(masks[i], args.output_mask_dir, rel_paths[i])
                
                record = {
                    "image_id": image_ids[i],
                    "img_path": rel_paths[i],
                    "caption": texts[i]
                }
                # Write as a single JSON line
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            # Force write to disk so we can inspect the file while the script is running
            out_f.flush()
            
    print(f"✅ Finished. Output successfully streamed to {args.output_file}")

if __name__ == "__main__":
    main()