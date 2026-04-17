import json

# 读取 output.json 文件
with open('/home/share/qrn8y2ug/home/ysuanAJ27/ljc/LAVIS-main/handle_caption/output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 创建一个新的字典来存储处理后的结果
processed_data = {}

# 处理每个句子，只保留键和英文翻译部分
for key, value in data.items():
    # 找到 'English (translation and summary):' 之后的英文翻译部分
    translation_start = value.find('<|im_end|>\n<|im_start|>assistant\n ')
    if translation_start != -1:
        english_translation = value[translation_start + len('<|im_end|>\n<|im_start|>assistant\n '):].strip()
        processed_data[key] = english_translation

# 将结果写回 JSON 文件
with open('/home/share/qrn8y2ug/home/ysuanAJ27/ljc/LAVIS-main/handle_caption/output-1.json', 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=4)

print("Processed output.json file saved.")
