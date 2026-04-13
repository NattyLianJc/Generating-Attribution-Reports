"""
 Training Entry for Manipulation Forgery Detection / Multimodal Tasks
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# It's better to avoid 'import *' in main entry scripts to keep namespace clean
import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.registry import registry
from lavis.common.utils import now

import lavis.datasets.builders as builders
import lavis.models as models
import lavis.processors as processors
import lavis.runners as runners
import lavis.tasks as tasks
import lavis.common.optims as optims

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    # 将 default 修改为新的路径
    parser.add_argument("--cfg-path", default="configs/caption_coco_ft.yaml", help="path to configuration file.")
    parser.add_argument("--options", nargs="+", help="override config settings.")
    return parser.parse_args()

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def apply_training_strategy(model, training_stage: str):
    """
    Dynamically freeze/unfreeze model parameters based on the training stage.
    """
    model.training_stage = training_stage
    
    if training_stage == "pretrain":
        # Stage 1: Contrastive / Q-Former pretraining
        for name, param in model.named_parameters():
            if any(key in name for key in [
                "Qformer", "qformer", "text_proj", "fc_layer", 
                "t5_proj", "mlm_head", "mask_decoder", 
                "prompt_proj", "cross_attention_layer"
            ]):
                param.requires_grad = True
            else:
                param.requires_grad = False
                
            # Extra safety for T5 core layers
            if "t5_model" in name and ("embed_tokens" in name or "lm_head" in name):
                param.requires_grad = False
                
    elif training_stage == "finetune":
        # Stage 2: Full finetuning
        for name, param in model.named_parameters():
            if "visual_encoder" in name and "Adapter" not in name:
                param.requires_grad = False
            elif "t5_model" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    else:
        raise ValueError(f"Unknown training stage: {training_stage}")
        
    # Log trainable parameters for debugging
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{training_stage.upper()} STAGE] Trainable parameters: {trainable_params:,}")


def main():
    job_id = now()
    print(f"Starting Job: {job_id}")
    
    args = parse_args()
    cfg = Config(args)

    # Safe device allocation using LAVIS dist_utils (handles single/multi-GPU gracefully)
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger()

    cfg.pretty_print()

    # Build task, datasets, and model using LAVIS registry pattern
    task = tasks.setup_task(cfg) 
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    # Apply freezing strategy driven by config (default to finetune if not set)
    stage = cfg.run_cfg.get("training_stage", "finetune")
    apply_training_strategy(model, stage)

    # Fetch runner and start training
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))
    runner = runner_cls(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    
    runner.train()

if __name__ == "__main__":
    main()