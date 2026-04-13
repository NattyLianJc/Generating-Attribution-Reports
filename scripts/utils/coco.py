from lavis.tasks.pycocoeval import COCOEvalCap
from pycocotools.coco import COCO

coco_gt_root = "/data/HOME/lly/workspace/dataset"
results_file = "/data/HOME/lly/workspace/dataset/val_epoch9.json"
split = "val"

filenames = {
        "val": "eval_gt_4cap.json",
    }

import json
import os
annotation_file = os.path.join(coco_gt_root, filenames[split])

coco = COCO(annotation_file)
coco_result = coco.loadRes(results_file)

coco_eval = COCOEvalCap(coco, coco_result)
coco_eval.evaluate()