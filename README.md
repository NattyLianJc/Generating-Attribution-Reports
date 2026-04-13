
# Generating Attribution Reports for Manipulated Facial Images

This is the official implementation of the paper:

**"Generating Attribution Reports for Manipulated Facial Images: A Dataset and Baseline"**

> **Authors:** [Jingchun Lian](mailto:15829901729@stu.xjtu.edu.cn)¹, Lingyu Liu¹, Yaxiong Wang², Yujiao Wu³, Lianwei Wu⁴, Li Zhu¹, and Zhedong Zheng⁵
> 
> ¹Xi'an Jiaotong University, ²Hefei University of Technology, ³CSIRO, ⁴Northwestern Polytechnical University, ⁵University of Macau

## 📖 Introduction

DFF (**D**eepFake Detection and **F**orensic Explanation **F**ramework) is a novel multimodal framework built upon the **InstructBLIP (Flan-T5 XL)** architecture. By integrating a multi-label visual classifier (**Face-ViT**) as an auxiliary perception module, DFF goes beyond binary classification to provide **explainable deepfake detection**:

1. **Precise Localization**: Generates binary masks to segment manipulated facial regions.
    
2. **Forensic Explanation**: Produces high-quality natural language reports explaining _why_ specific areas are identified as unnatural.
    

Our framework is optimized for high-performance computing (e.g., NVIDIA H100/A100), supporting distributed training and asynchronous data pipelines.

---

## 🛠️ 1. Environment Setup

We recommend using Conda to manage your environment:

Bash

```
# 1. Create and activate the environment
conda create -n dff_env python=3.10 -y
conda activate dff_env

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Download necessary NLP linguistic resources
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt')"
```

---

## 📦 2. Dataset: MMTT (Multi-Modal Tamper Tracing)

The **MMTT Dataset** is hosted on Hugging Face.

🔗 **Link**: [LianJC/MMTT-Dataset](https://huggingface.co/datasets/LianJC/MMTT-Dataset)

### Data Reconstruction

Create a `dataset/` directory in the project root and organize it as follows:

Bash

```
mkdir dataset && cd dataset
# Merge split volumes and decompress
cat dataset.tar.gz.part* > dataset_merged.tar.gz
tar -xzvf dataset_merged.tar.gz
tar -xzvf dataset_json.tar.gz
```

**Required Directory Structure:**

Plaintext

```
/dff/
└── dataset/
    ├── dataset/                    # Image root
    │   ├── celeb512/               # CelebA-HQ based manipulations
    │   ├── face_attribute_image/   # Attribute feature sets
    │   └── FFHQ512/                # FFHQ based manipulations
    ├── train-mutilabel-simple.json # Standardized training annotations
    └── eval-mutilabel-simple.json  # Validation annotations
```

---

## 📥 3. Model Checkpoints

### Pre-trained Backbones

Place the following base models in the `model_parameter/` directory:

- **LLM**: [flan-t5-xl](https://huggingface.co/google/flan-t5-xl)
    
- **BERT**: [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)
    
- **EVA-ViT**: [eva_vit_g.pth](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth)
    
- **InstructBLIP**: [instruct_blip_flanxl_trimmed.pth](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_flanxl_trimmed.pth)
    

### Fine-tuned Models (Our Baseline)

|**Module**|**Description**|**Repository**|
|---|---|---|
|**Face-ViT**|Multi-label region classifier (ViT-H/14)|[Face-ViT-MultiLabel](https://huggingface.co/LianJC/Face-ViT-MultiLabel)|
|**DFF-Model**|Integrated Detection & Explanation Model|[DFF-InstructBLIP-Detection](https://huggingface.co/LianJC/DFF-InstructBLIP-Detection)|

> ⚠️ **Configuration Notice**: Ensure `qformer_text_input` is set to `True` in `lavis/configs/models/blip2/blip2_instruct_flant5xl.yaml`.


You can download all required weights with a single command: 

```bash
python scripts/utils/download_weights.py
```

---

## 🚀 4. Training

We use `torchrun` for distributed training. Configuration is managed via YAML files in the `configs/` directory.

Bash

```
# Distributed Training (e.g., 8 GPUs)
torchrun --nproc_per_node=8 train.py --cfg-path configs/caption_coco_ft.yaml
```

**Training Notes:**

- **Loss Scaling**: You can adjust the weight of the Gate Loss in `lavis/models/blip2_models/blip2_t5_instruct.py` (e.g., `loss = lm_loss + seg_loss + 0.5 * gate_loss`).
    
- **Checkpoints**: Saving paths are defined by the `output_dir` field in the config YAML.
    

---

## 🧠 5. Inference and Evaluation

Our pipeline separates result generation and metric calculation for maximum performance.

### High-Performance Inference

Use the refactored evaluation script for batched prediction (Captions & Masks):

Bash

```
python scripts/evaluation/evaluate.py \
    --checkpoint model_parameter/2111_checkpoint_best.pth \
    --json_file dataset/eval-mutilabel-simple.json \
    --output_json results/test_captions.json \
    --output_mask_dir results/test_masks \
    --batch_size 16 \
    --num_workers 8
```

### Academic Metric Calculation

Compute NLP (CIDEr, BLEU, ROUGE) and Vision (mIoU, Precision, Recall) scores using standardized tokenizers:

Bash

```
python scripts/evaluation/evaluate_metrics.py \
    --gt_json dataset/eval-mutilabel-simple.json \
    --pred_json results/test_captions.json \
    --gt_mask_base dataset \
    --pred_mask_base results/test_masks \
    --output_file results/final_metrics.json
```

---

## 📜 Citation

If you find our work useful, please cite:

```
@article{lian2026generating,
  title={Generating Attribution Reports for Manipulated Facial Images: A Dataset and Baseline},
  author={Lian, Jingchun and Liu, Lingyu and Wang, Yaxiong and Wu, Yujiao and Wu, Lianwei and Zhu, Li and Zheng, Zhedong},
  journal={arXiv preprint},
  year={2026}
}
```