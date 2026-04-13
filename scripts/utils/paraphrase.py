

import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. 在程序开始时就加载 Flan-T5-XL
model_name = "/home/ubuntu/ljc/LAVIS-main/model_parameter/flant5xl"
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("Model loaded successfully.\n")


def paraphrase_caption(original_text: str) -> str:
    """
    Uses Flan-T5-XL to paraphrase the given English text.
    """
    prompt = f"Paraphrase the following sentence in English with the same meaning:\n{original_text}\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.7,  # can adjust for more or less creativity
            top_k=50,
            do_sample=True
        )

    paraphrased_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return paraphrased_text


def main():
    # 2. 指定输入/输出文件夹
    input_folder = "/home/ubuntu/ljc/LAVIS-main/dataset/input"
    output_folder = "/home/ubuntu/ljc/LAVIS-main/dataset/output"
    os.makedirs(output_folder, exist_ok=True)

    # 3. 获取所有 JSON 文件
    json_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

    # 使用 tqdm 显示处理 JSON 文件的进度
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        input_path = os.path.join(input_folder, json_file)
        output_path = os.path.join(output_folder, json_file)

        # 3.1 读取 JSON
        with open(input_path, "r", encoding="utf-8") as f_in:
            data = json.load(f_in)
            # 假设顶层是一个列表，每个元素形如：
            # {
            #   "image_id": "...",
            #   "caption": "...",
            #   "img_path": "...",
            #   "mask_path": "...",
            #   "instance_id": "...",
            #   "multilabel": [...]
            # }

        # 3.2 遍历每条记录，改写 caption
        # 在遍历 data 的过程中也加一个 tqdm 进度条
        for idx in tqdm(range(len(data)), desc=f"Processing records in {json_file}", leave=False):
            item = data[idx]
            original_caption = item.get("caption", "")
            if original_caption.strip():
                new_caption = paraphrase_caption(original_caption)
            else:
                new_caption = original_caption
            item["caption"] = new_caption

        # 3.3 将更新后的数据写入新的 JSON 文件
        with open(output_path, "w", encoding="utf-8") as f_out:
            json.dump(data, f_out, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()

