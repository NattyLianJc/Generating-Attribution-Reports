# Generating Attribution Reports for Manipulated Facial Images

[](https://2026.aclweb.org/)
[](https://opensource.org/licenses/BSD-3-Clause)

This is the official implementation of the paper:

**"Generating Attribution Reports for Manipulated Facial Images: A Dataset and Baseline"**

> **Authors:** Jingchun Lian¹, Lingyu Liu¹, Yaxiong Wang², Yujiao Wu³, Lianwei Wu⁴, Li Zhu¹, and Zhedong Zheng⁵  
> ¹Xi'an Jiaotong University, ²Hefei University of Technology, ³CSIRO, ⁴Northwestern Polytechnical University, ⁵University of Macau

## 📖 Introduction

DFF (**D**eepFake Detection and **F**orensic Explanation **F**ramework) is a novel multimodal framework built upon the **InstructBLIP (Flan-T5 XL)** architecture. By integrating a multi-label visual classifier (**Face-ViT**) as an auxiliary perception module, DFF goes beyond binary classification to provide **explainable deepfake detection**:

1.  **Precise Localization**: Generates binary masks to segment manipulated facial regions.
2.  **Forensic Explanation**: Produces high-quality natural language reports explaining *why* specific areas are identified as unnatural.

Our framework is optimized for high-performance computing (e.g., NVIDIA H100/A100), supporting distributed training and asynchronous data pipelines.

-----

## 📊 Performance (Validation & Test Sets)

In our paper, we primarily report the validation metrics for standard comparison with state-of-the-art baselines. To provide a comprehensive view of DFF's robust generalization capabilities, we present our model's performance on both the Validation Set and the massive unseen **Test Set (21,679 samples)** below:

| Category | Metric | Validation Set (%) | Test Set (%) |
| :--- | :--- | :---: | :---: |
| **Report Generation** | CIDEr | 59.30 | **69.48** |
| *(NLP)* | ROUGE-L | 28.80 | **30.54** |
| | BLEU-4 | 12.50 | **13.47** |
| | BLEU-3 | 16.00 | **16.73** |
| | BLEU-2 | 22.10 | **22.49** |
| | BLEU-1 | **35.00** | 34.50 |
| **Forgery Localization** | IoU | **73.67** | 66.06 |
| *(Vision)* | Precision | **91.43** | 76.27 |
| | Recall | **86.22** | 81.83 |

> *Note: NLP metrics are evaluated using the standard PTBTokenizer. All scores are scaled by 100 for readability. Remarkably, the model demonstrates exceptionally strong generalization in text generation on the unseen test set, while maintaining robust localization performance.*

-----

## 🛠️ 1. Environment Setup

We recommend using Conda to manage your environment:

```bash
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

-----

## 📦 2. Dataset: MMTT (Multi-Modal Tamper Tracing)

The **MMTT Dataset** is hosted on Hugging Face.

🔗 **Link**: [LianJC/MMTT-Dataset](https://huggingface.co/datasets/LianJC/MMTT-Dataset)

### Data Reconstruction

Create a `dataset/` directory in the project root and organize it as follows:

```bash
mkdir dataset && cd dataset
# Merge split volumes and decompress
cat dataset.tar.gz.part* > dataset_merged.tar.gz
tar -xzvf dataset_merged.tar.gz
tar -xzvf dataset_json.tar.gz
```

**Required Directory Structure:**

```text
/dff/
└── dataset/
    ├── dataset/                    # Image root
    │   ├── celeb512/               # CelebA-HQ based manipulations
    │   ├── face_attribute_image/   # Attribute feature sets
    │   └── FFHQ512/                # FFHQ based manipulations
    ├── train-mutilabel-simple.json # Standardized training annotations
    └── eval-mutilabel-simple.json  # Validation annotations
```

-----

## 📥 3. Model Checkpoints

### Pre-trained Backbones

Place the following base models in the `model_parameter/` directory:

  - **LLM**: [flan-t5-xl](https://huggingface.co/google/flan-t5-xl)
  - **BERT**: [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)
  - **EVA-ViT**: [eva\_vit\_g.pth](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth)
  - **InstructBLIP**: [instruct\_blip\_flanxl\_trimmed.pth](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_flanxl_trimmed.pth)

### Fine-tuned Models (Our Baseline)

| Module | Description | Repository |
| :--- | :--- | :--- |
| **Face-ViT** | Multi-label region classifier (ViT-H/14) | [Face-ViT-MultiLabel](https://huggingface.co/LianJC/Face-ViT-MultiLabel) |
| **DFF-Model**| Integrated Detection & Explanation Model | [DFF-InstructBLIP-Detection](https://huggingface.co/LianJC/DFF-InstructBLIP-Detection) |

> ⚠️ **Configuration Notice**: Ensure `qformer_text_input` is set to `True` in `lavis/configs/models/blip2/blip2_instruct_flant5xl.yaml`.

You can download all required weights automatically with a single command:

```bash
python scripts/utils/download_weights.py
```

-----

## 🚀 4. Training

We use `torchrun` for distributed training. Configuration is managed via YAML files in the `configs/` directory.

```bash
# Distributed Training (e.g., 8 GPUs)
torchrun --nproc_per_node=8 train.py --cfg-path configs/caption_coco_ft.yaml
```

**Training Notes:**

  - **Checkpoints**: Saving paths are defined by the `output_dir` field in the config YAML.

-----

## 🧠 5. Inference and Evaluation

Our pipeline separates result generation and metric calculation for maximum performance.

### High-Performance Inference

Use the refactored evaluation script for batched prediction (Captions & Masks). This outputs a `.jsonl` file for real-time streaming validation.

```bash
python scripts/evaluation/evaluate.py \
    --checkpoint model_parameter/2111_checkpoint_best.pth \
    --json_file dataset/test-mutilabel.json \
    --output_file results/test_captions.jsonl \
    --output_mask_dir results/test_masks \
    --batch_size 16 \
    --num_workers 8
```

### Academic Metric Calculation

Compute NLP (CIDEr, BLEU, ROUGE) and Vision (mIoU, Precision, Recall) scores:

```bash
python scripts/evaluation/evaluate_metrics.py \
    --gt_json dataset/test-mutilabel.json \
    --pred_json results/test_captions.jsonl \
    --gt_mask_base dataset \
    --pred_mask_base results/test_masks \
    --output_file results/final_metrics.json
```

-----

## 📜 Citation

If you find our work or dataset useful, please cite our paper. *(Accepted to ACL 2026 Main Track)*

```bibtex
@inproceedings{lian2026generating,
  title={Generating Attribution Reports for Manipulated Facial Images: A Dataset and Baseline},
  author={Lian, Jingchun and Liu, Lingyu and Wang, Yaxiong and Wu, Yujiao and Wu, Lianwei and Zhu, Li and Zheng, Zhedong},
  booktitle={Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026},
  note={To appear}
}
```