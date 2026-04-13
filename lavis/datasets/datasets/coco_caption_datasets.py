"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from PIL import Image, ImageFile
from lavis.datasets.datasets.caption_datasets import (
    CaptionDataset, 
    CaptionEvalDataset, 
    ManipulationMixin
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

COCOCapDataset = CaptionDataset

class COCOCapEvalDataset(CaptionEvalDataset, ManipulationMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        # 1. Base loading logic from parent
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["img_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])

        # 2. Add Manipulation Data logic via Mixin (DRY)
        mask_source_path = os.path.join(self.vis_root, ann["mask_path"])
        image_size = image.shape[1]
        mask = self._load_and_process_mask(mask_source_path, image_size)
        multilabel = self._get_multilabel_tensor(ann)

        return {
            "image": image,
            "image_id": ann["image_id"],
            "caption": caption,
            "pt": "",
            "mask": mask,
            "instance_id": ann["instance_id"],
            "multilabel": multilabel,
        }


class NoCapsEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        return {
            "image": image,
            "image_id": ann["img_id"],
            "instance_id": ann["instance_id"],
        }