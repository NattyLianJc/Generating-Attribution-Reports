"""
 Batch evaluation script to process multiple JSON files in a directory.
 Reuses high-performance DataLoader logic.
"""

import json
import os
import glob
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from lavis.models import load_model_and_preprocess

# Import shared components from the primary evaluate script
# Ensure the root directory is in PYTHONPATH
from scripts.evaluation.evaluate import DeepFakeEvalDataset, save_mask

def main():
    parser = argparse.ArgumentParser(description="Evaluate a directory of JSON files.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_dir", type=str, default="classified_json_output")
    parser.add_argument("--output_base_dir", type=str, default="inference_results")
    parser.add_argument("--image_dir", type=str, default="dataset")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup model once for all files
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5_instruct", model_type="flant5xl", is_eval=True, device=device
    )
    model.load_checkpoint(args.checkpoint)
    model.eval()

    # Find all target JSON files
    json_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    if not json_files:
        print(f"No JSON files found in {args.input_dir}")
        return

    print(f"📂 Found {len(json_files)} files in queue.")

    for json_path in json_files:
        file_name = os.path.basename(json_path)
        file_prefix = os.path.splitext(file_name)[0]
        
        output_json = os.path.join(args.output_base_dir, f"{file_prefix}_captions.json")
        output_mask_dir = os.path.join(args.output_base_dir, f"{file_prefix}_masks")
        
        print(f"\n--- Current Task: {file_name} ---")
        
        with open(json_path, "r") as f:
            data = json.load(f)
        if not data: continue

        # Performance boost: Use DataLoader even for single file iterations
        dataset = DeepFakeEvalDataset(data, args.image_dir, vis_processors)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)
        
        file_results = []
        for batch in tqdm(dataloader, desc=f"Predicting {file_prefix}"):
            images = batch["image"].to(device, non_blocking=True)
            with torch.no_grad():
                texts, masks = model.generate({"image": images}, return_seg=True)
            
            for i in range(len(batch["image_id"])):
                save_mask(masks[i], output_mask_dir, batch["relative_path"][i])
                file_results.append({
                    "image_id": batch["image_id"][i],
                    "caption": texts[i]
                })

        with open(output_json, "w") as f:
            json.dump(file_results, f, indent=4)

    print("\n🎉 Batch processing finished successfully.")

if __name__ == "__main__":
    main()