# # Install transformers from source - only needed for versions <= v4.34
# # pip install git+https://github.com/huggingface/transformers.git
# # pip install accelerate

# import torch
# from transformers import pipeline

# pipe = pipeline("text-generation", model="/home/share/qrn8y2ug/home/ysuanAJ27/ljc/LAVIS-main/model-parameter/TowerInstruct-7B-v0.2", torch_dtype=torch.float32, device_map="auto")
# # We use the tokenizer’s chat template to format each message - see https://hf-mirror.com/docs/transformers/main/en/chat_templating
# messages = [
#     {"role": "user", "content": "Translate and summarize the following sentence into English within 60 words using single words and adjectives to describe features, and combine sentences: 这名女子左边眉毛部分缺失，瞳孔大小不一致。眼睛变形，嘴巴变形。上嘴唇没有纹理，眼睫毛轮廓不清晰，头发部分缺失，牙齿轮廓不清晰，脖子肤色和面部肤色不一致."},
# ]
# prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# outputs = pipe(prompt, max_new_tokens=80, do_sample=False)
# print(outputs[0]["generated_text"])
# # <|im_start|>user
# # Translate the following text from Portuguese into English.
# # Portuguese: Um grupo de investigadores lançou um novo modelo para tarefas relacionadas com tradução.
# # English:<|im_end|>
# # <|im_start|>assistant
# # A group of researchers has launched a new model for translation-related tasks.

import json
import torch
from transformers import pipeline
from tqdm import tqdm

# 初始化模型
pipe = pipeline("text-generation", model="/home/share/qrn8y2ug/home/ysuanAJ27/ljc/LAVIS-main/model-parameter/TowerInstruct-7B-v0.2", torch_dtype=torch.float32, device_map="auto")

def translate_and_summarize(text):
    messages = [
        {"role": "user", "content": f"Translate and abbreviate the following sentence into English within 60 words using single words and adjectives to describe features, and combine sentences.\nChinese: {text}\nEnglish:"}
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=160, do_sample=False)
    generated_text = outputs[0]["generated_text"]
    # 提取英文翻译部分
    translation = generated_text.split('<|im_end|>\n<|im_start|>assistant\n ')[-1].strip()
    return translation

# 读取 JSON 文件
with open('/home/share/qrn8y2ug/home/ysuanAJ27/ljc/LAVIS-main/handle_caption/0726-input.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 处理每个句子
translated_data = {}
for key, sentence in tqdm(data.items(), desc="Translating and summarizing"):
    translated_data[key] = translate_and_summarize(sentence)

# 将结果写回 JSON 文件
with open('/home/share/qrn8y2ug/home/ysuanAJ27/ljc/LAVIS-main/handle_caption/0726-output.json', 'w', encoding='utf-8') as f:
    json.dump(translated_data, f, ensure_ascii=False, indent=4)

print("Translation and summarization complete. Results saved to output.json.")
