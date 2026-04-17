"""
 Comprehensive evaluation script for both Vision (Mask IoU/Prec/Rec) 
 and NLP (BLEU/ROUGE/CIDEr) metrics.
 Unified script to replace all previous individual compute scripts.
"""

import json
import os
import argparse
import numpy as np
import torch
import concurrent.futures
from PIL import Image
from tqdm import tqdm

# Standard NLP evaluation packages
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

def evaluate_masks(pred_path, gt_path):
    """
    Calculates IoU, Precision, and Recall for a single pair of masks.
    
    Args:
        pred_path (str): Path to the predicted binary mask.
        gt_path (str): Path to the ground truth binary mask.
    Returns:
        tuple: (iou, precision, recall) or None if error occurs.
    """
    try:
        # Load ground truth
        gt_img = Image.open(gt_path).convert('L')
        target_size = gt_img.size
        gt = np.array(gt_img) > 127
        
        # Load prediction and resize to match GT using NEAREST to preserve binary edges
        pred_img = Image.open(pred_path).convert('L')
        pred_img = pred_img.resize(target_size, Image.NEAREST)
        pred = np.array(pred_img) > 127
        
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        pred_sum = pred.sum()
        gt_sum = gt.sum()

        # Calculation with safety checks for zero-division
        iou = 1.0 if union == 0 else float(intersection / union)
        precision = float(intersection / pred_sum) if pred_sum > 0 else (1.0 if gt_sum == 0 else 0.0)
        recall = float(intersection / gt_sum) if gt_sum > 0 else (1.0 if pred_sum == 0 else 0.0)

        return iou, precision, recall
    except Exception as e:
        return None

def worker_evaluate_single(img_id, gt_item, pred_item, gt_mask_base, pred_mask_base):
    """
    Worker function for parallel process pool.
    """
    # Map paths correctly based on your JSON structure
    img_rel_path = gt_item.get('img_path') or gt_item.get('relative_path')
    mask_rel_path = gt_item.get('mask_path')
    
    # In some versions, pred mask is saved with the same name as original image but in PNG
    pred_mask_filename = os.path.splitext(img_rel_path)[0] + ".png"
    pred_mask_path = os.path.join(pred_mask_base, pred_mask_filename)
    gt_mask_path = os.path.join(gt_mask_base, mask_rel_path)

    if not (os.path.exists(pred_mask_path) and os.path.exists(gt_mask_path)):
        return None
    
    return evaluate_masks(pred_mask_path, gt_mask_path)

def evaluate_text(gts, res):
    """
    Calculates standard NLP metrics using PTB Tokenizer.
    
    Args:
        gts (dict): Dictionary of ground truth captions.
        res (dict): Dictionary of predicted captions.
    Returns:
        dict: Calculated metrics (BLEU, ROUGE, CIDEr).
    """
    print("\n[Text] Tokenizing using standard PTBTokenizer...")
    tokenizer = PTBTokenizer()
    gts_tok = tokenizer.tokenize(gts)
    res_tok = tokenizer.tokenize(res)

    print("[Text] Calculating NLP Metrics...")
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    eval_results = {}
    for scorer, method in scorers:
        score, _ = scorer.compute_score(gts_tok, res_tok)
        if isinstance(method, list):
            for sc, m in zip(score, method):
                eval_results[m] = float(sc)
        else:
            eval_results[method] = float(score)
            
    return eval_results

def main():
    parser = argparse.ArgumentParser(description="Consolidated Metrics Evaluation Script")
    parser.add_argument("--gt_json", type=str, required=True, help="Ground truth JSON file.")
    parser.add_argument("--pred_json", type=str, required=True, help="Generated results JSON file.")
    parser.add_argument("--gt_mask_base", type=str, default="dataset", help="Root dir for GT masks.")
    parser.add_argument("--pred_mask_base", type=str, required=True, help="Dir where predicted masks are saved.")
    parser.add_argument("--output_file", type=str, default="results/final_metrics.json")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    # 1. Load Data
    with open(args.gt_json, 'r') as f: gt_data = json.load(f)
    with open(args.pred_json, 'r', encoding='utf-8') as f:
        if args.pred_json.endswith('.jsonl'):
            pred_data = [json.loads(line) for line in f]
        else:
            pred_data = json.load(f)

    gt_dict = {item['image_id']: item for item in gt_data}
    pred_dict = {item['image_id']: item for item in pred_data}
    common_ids = list(set(gt_dict.keys()).intersection(set(pred_dict.keys())))

    print(f"Total Common Samples: {len(common_ids)}")

    gts_text, res_text = {}, {}
    total_iou, total_precision, total_recall = 0.0, 0.0, 0.0
    valid_mask_count = 0

    # 2. Parallel Vision Evaluation
    print(f"🚀 Using {args.num_workers} CPU cores for vision evaluation...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                worker_evaluate_single, 
                idx, gt_dict[idx], pred_dict[idx], args.gt_mask_base, args.pred_mask_base
            ): idx for idx in common_ids
        }

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(common_ids), desc="Parallel Eval"):
            img_id = futures[future]
            
            # Format text for tokenizer: {id: [{'caption': '...'}]}
            gts_text[img_id] = [{'caption': gt_dict[img_id]['caption']}]
            res_text[img_id] = [{'caption': pred_dict[img_id]['caption']}]
            
            res = future.result()
            if res:
                total_iou += res[0]
                total_precision += res[1]
                total_recall += res[2]
                valid_mask_count += 1

    # 3. Aggregation
    mIoU = total_iou / valid_mask_count if valid_mask_count > 0 else 0
    mPrecision = total_precision / valid_mask_count if valid_mask_count > 0 else 0
    mRecall = total_recall / valid_mask_count if valid_mask_count > 0 else 0

    # 4. Text Evaluation
    text_metrics = evaluate_text(gts_text, res_text)

    # 5. Finalize Results
    final_results = {
        "metadata": {
            "evaluated_samples": len(common_ids),
            "matched_masks": valid_mask_count
        },
        "vision_metrics": {
            "mIoU": round(mIoU, 6),
            "precision": round(mPrecision, 6),
            "recall": round(mRecall, 6)
        },
        "nlp_metrics": {k: round(v, 6) for k, v in text_metrics.items()}
    }

    # 6. Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print("\n" + "="*60)
    print(f"📊 Final Results saved to: {args.output_file}")
    print(f"mIoU: {mIoU:.4f} | CIDEr: {text_metrics['CIDEr']:.4f} | Bleu_4: {text_metrics['Bleu_4']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()