"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import logging
import os
import numpy as np
import torch
import torch.distributed as dist
from torch.autograd import Function

# Import both the decorator (main_process) and the checker (is_main_process)
from lavis.common.dist_utils import main_process, is_main_process, is_dist_avail_and_initialized
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.tasks.pycocoeval import COCOEvalCap
from pycocotools.coco import COCO


def iou(outputs: np.array, labels: np.array):
    """Calculate IoU for numpy arrays"""
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou.mean()


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    @staticmethod
    def forward(ctx, input, target):
        eps = 0.0001
        inter = torch.dot(input.view(-1), target.view(-1))
        union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * inter.float() + eps) / union.float()
        ctx.save_for_backward(input, target)
        ctx.inter = inter
        ctx.union = union
        return t

    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_tensors
        inter = ctx.inter
        union = ctx.union
        grad_input = grad_target = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * union - inter) / (union * union)

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    s = torch.zeros(1, device=input.device)

    for i, (inp, tar) in enumerate(zip(input, target)):
        s = s + DiceCoeff.apply(inp, tar)

    return s / (i + 1)


def maybe_generate_coco_gt(annotation_path):
    """
    Checks if a COCO-format ground truth file exists. 
    If not, generates it from the standard dataset JSON.
    Uses barrier synchronization for distributed environments.
    """
    if not annotation_path or not os.path.exists(annotation_path):
        return None
    
    gt_path = annotation_path.replace(".json", "_gt.json")
    
    # Use is_main_process() for boolean check
    if is_main_process():
        if not os.path.exists(gt_path):
            logging.info(f"Ground truth file not found. Generating: {gt_path}")
            try:
                with open(annotation_path, 'r') as f:
                    data = json.load(f)
                
                new_data = {'annotations': [], 'images': []}
                image_ids = set()
                
                for item in data:
                    if 'instance_id' in item and 'image_id' in item:
                        new_data['annotations'].append({
                            'image_id': item['image_id'],
                            'caption': item['caption'],
                            'id': item['instance_id']
                        })
                        if item['image_id'] not in image_ids:
                            new_data['images'].append({'id': item['image_id']})
                            image_ids.add(item['image_id'])
                    else:
                        logging.warning(f"Skipping item due to missing ID in {annotation_path}")
                
                with open(gt_path, 'w') as f:
                    json.dump(new_data, f, indent=4)
                logging.info(f"Successfully generated GT file: {gt_path}")
            except Exception as e:
                logging.error(f"Failed to generate GT file: {e}")

    # Synchronize all processes
    if is_dist_avail_and_initialized():
        dist.barrier()
            
    return gt_path if os.path.exists(gt_path) else None


@registry.register_task("captioning")
class CaptionTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True, gt_filenames=None):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate
        self.report_metric = report_metric
        self.gt_filenames = gt_filenames or {}

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        dataset_cfg = list(cfg.datasets_cfg.values())[0]

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate
        report_metric = run_cfg.get("report_metric", True)

        build_info = dataset_cfg.get("build_info", {})
        splits = ["val", "test"]
        gt_filenames = {}
        
        for split in splits:
            raw_ann_path = build_info.get("annotations", {}).get(split, {}).get("storage")
            if raw_ann_path:
                gt_path = maybe_generate_coco_gt(raw_ann_path)
                if gt_path:
                    gt_filenames[split] = gt_path

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
            gt_filenames=gt_filenames
        )

    def valid_step(self, model, samples):
        results = []

        if model.training_stage == "pretrain":
            loss_output = model.forward_pretrain(samples)
            loss_dict = {"contrastive_loss": loss_output["loss"].item()}
            return results, loss_output["loss"], loss_dict

        elif model.training_stage == "finetune":
            captions = model.generate(
                samples,
                use_nucleus_sampling=False,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
            )

            loss_output = model(samples)
            loss_dict = {k: v for k, v in loss_output.items() if "loss" in k}
            
            img_ids = samples["image_id"]
            for caption, img_id in zip(captions, img_ids):
                results.append({"caption": caption, "image_id": img_id})

            # Evaluate Segmentation Metrics
            thresholds = (0.1, 0.3, 0.5, 0.7, 0.9)
            eiou, edice, eprecision, erecall = 0, 0, 0, 0
            
            true_mask_p = samples["mask"].squeeze(dim=1)
            pred = loss_output["seg_out"]

            for th in thresholds:
                gt_vmask_p = (true_mask_p > th).float()
                vpred = (pred > th).float()

                eiou += iou(vpred.cpu().numpy().astype('int32'), gt_vmask_p.cpu().numpy().astype('int32'))
                edice += dice_coeff(vpred, gt_vmask_p).item()

                tp = (vpred * gt_vmask_p).sum().item()
                fp = (vpred * (1 - gt_vmask_p)).sum().item()
                fn = ((1 - vpred) * gt_vmask_p).sum().item()

                eprecision += tp / (tp + fp + 1e-8)
                erecall += tp / (tp + fn + 1e-8)

            num_thresholds = len(thresholds)
            loss_dict.update({
                "eiou": eiou / num_thresholds,
                "edice": edice / num_thresholds,
                "precision": eprecision / num_thresholds,
                "recall": erecall / num_thresholds
            })

            return results, loss_output["loss"], loss_dict
        else:
            raise ValueError(f"Unknown training stage: {model.training_stage}")

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="image_id"
        )

        if not val_result or "caption" not in val_result[0] or split_name not in self.gt_filenames:
            return {"agg_metrics": 0.0}

        return self._report_metrics(eval_result_file, split_name) if self.report_metric else {"agg_metrics": 0.0}

    @main_process  # Use main_process correctly as a decorator
    def _report_metrics(self, eval_result_file, split_name):
        gt_file = self.gt_filenames.get(split_name)
        if not gt_file:
            return {"agg_metrics": 0.0}

        coco_val = coco_caption_eval(gt_file, eval_result_file)

        agg_metrics = coco_val.eval["CIDEr"] + coco_val.eval["Bleu_4"]
        log_stats = {split_name: {k: v for k, v in coco_val.eval.items()}}

        with open(os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in coco_val.eval.items()}
        coco_res["agg_metrics"] = agg_metrics
        return coco_res


def coco_caption_eval(annotation_file, results_file):
    """Standard COCO evaluation wrapper."""
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.evaluate()

    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval