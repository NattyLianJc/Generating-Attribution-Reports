"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from lavis.datasets.datasets.base_dataset import BaseDataset


class ManipulationMixin:
    """
    Mixin class providing shared logic for loading and processing 
    manipulation masks and multilabel annotations.
    """
    def _load_and_process_mask(self, mask_path, target_size):
        """
        Reads a mask image, converts it to binary format, and resizes it.
        """
        # Step 1: Read image and convert to grayscale
        mask_image = Image.open(mask_path).convert('L')
        mask_np = np.array(mask_image)

        # Step 2: Create binary mask (0 or 255)
        binary_mask_np = (mask_np > 0).astype(np.uint8) * 255
        mask_pil = Image.fromarray(binary_mask_np)

        # Step 3: Define and apply transforms (using NEAREST interpolation for masks)
        trans_msk = transforms.Compose([
            transforms.Resize(
                (target_size, target_size), 
                interpolation=InterpolationMode.NEAREST
            ),
            transforms.ToTensor(),
        ])
        
        mask = trans_msk(mask_pil)
        return (mask > 0.5).float()

    def _get_multilabel_tensor(self, ann):
        """
        Extracts multilabel array and returns as a float32 tensor.
        """
        return torch.tensor(ann.get("multilabel", []), dtype=torch.float32)


class CaptionDataset(BaseDataset, ManipulationMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):
        ann = self.annotation[index]

        # Process Image
        image_path = os.path.join(self.vis_root, ann["img_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        # Process Caption
        caption = self.text_processor(ann["caption"])

        # Process Mask (Using Mixin)
        mask_source_path = os.path.join(self.vis_root, ann["mask_path"])
        image_size = image.shape[1]
        mask = self._load_and_process_mask(mask_source_path, image_size)

        # Process Multilabel (Using Mixin)
        multilabel = self._get_multilabel_tensor(ann)

        return {
            "image": image,
            "caption": caption,
            "pt": "",
            "mask": mask,
            "multilabel": multilabel,
        }


class CaptionEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["img_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])

        return {
            "image": image,
            "image_id": ann["image_id"],
            "caption": caption,
            "pt": "",
            "instance_id": ann["instance_id"],
        }