import json
import re
import os

# 定义同义词字典
facial_features = {
    "eye": ["eye", "eyes", "ocular", "optic", "vision"],
    "pupil": ["pupil", "pupils"],
    "lip": ["lip", "lips", "labium", "labia"],
    "mouth": ["mouth", "mouths", "oral cavity", "buccal"],
    "nose": ["nose", "noses", "nasal", "nostril", "nostrils"],
    "ear": ["ear", "ears", "auricle", "pinna", "pinnae"],
    "cheek": ["cheek", "cheeks", "buccal", "malar"],
    "forehead": ["forehead", "frontal", "brow", "brows"],
    "chin": ["chin", "chins", "mentum"],
    "jaw": ["jaw", "jaws", "mandible", "maxilla"],
    "eyebrow": ["eyebrow", "eyebrows"],
    "tooth": ["tooth", "teeth", "dentition"],
    "tongue": ["tongue", "tongues", "glossa", "lingua"],
    "beard": ["beard", "beards", "facial hair"],
    "mustache": ["mustache", "mustaches", "moustache", "moustaches"],
    "skin": ["skin", "derma", "epidermis", "cutis"],
    "hair": ["hair", "hairs", "locks", "tresses", "strands"],
    "wrinkle": ["wrinkle", "wrinkles", "line", "lines", "crease", "creases"],
    "freckle": ["freckle", "freckles", "lentigo", "lentigines"],
    "scar": ["scar", "scars", "cicatrix", "cicatrices"],
    "dimple": ["dimple", "dimples", "fossa", "fossae"]
}

# 标签顺序，列表顺序即为 multilabel 的下标位置
labels_order = list(facial_features.keys())
print("标签顺序:", labels_order)

# 加载其他三个 JSON 文件的已有 instance_id（文件路径根据实际情况修改）
used_ids = set()
other_files = ["dataset/eval-mutilabel.json", "dataset/train-mutilabel.json", "dataset/test-mutilabel.json"]
for filename in other_files:
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            try:
                data_temp = json.load(f)
                for item in data_temp:
                    if "instance_id" in item:
                        used_ids.add(item["instance_id"])
            except json.JSONDecodeError:
                continue


# 提取关键词的函数
def extract_keywords(caption):
    keywords = []
    for feature, synonyms in facial_features.items():
        for synonym in synonyms:
            if re.search(r'\b' + re.escape(synonym) + r'\b', caption, re.IGNORECASE):
                keywords.append(feature)
                break
    return keywords


# 将关键词列表转换为多标签向量
def keywords_to_multilabel(keywords):
    label_vector = [0] * len(labels_order)
    for keyword in keywords:
        if keyword in labels_order:
            idx = labels_order.index(keyword)
            label_vector[idx] = 1
    return label_vector


# 生成不重复的 instance_id（5位数字字符串）
def generate_unique_id(existing_ids, start=1):
    while True:
        candidate = f"{start:05d}"
        if candidate not in existing_ids:
            existing_ids.add(candidate)
            return candidate
        start += 1


# 读取原始 JSON 文件
with open('dataset/face_attribute_dataset.json', 'r') as file:
    data = json.load(file)

processed_data = []
id_counter = 1  # 可从1开始计数（也可根据需要调整起始值）

for item in data:
    # 修改 img_path 与 mask_path（在原有路径前拼接 "dataset/face_attribute_image/"）
    item["img_path"] = os.path.join("dataset/face_attribute_image", item["img_path"])
    item["mask_path"] = os.path.join("dataset/face_attribute_image", item["mask_path"])

    # 提取关键词并生成 multilabel 向量
    keywords = extract_keywords(item["caption"])
    multilabel = keywords_to_multilabel(keywords)

    # 如果 attribute 为 "surprised" 或 "angry"，特殊处理：
    # 需要确保 "eye", "eyebrow" 标签为 1，且将 "lip" 与 "mouth" 两个标签都设置为 1（代表 lip 加 mouth）
    if item["attribute"].lower() in ["surprised", "angry"]:
        for key in ["eye", "eyebrow", "lip", "mouth"]:
            if key in labels_order:
                index = labels_order.index(key)
                multilabel[index] = 1
    item["multilabel"] = multilabel

    # 生成唯一的 instance_id 与 image_id
    new_id = generate_unique_id(used_ids, start=id_counter)
    # 为避免重复，自增计数器（注意：生成后也更新计数器）
    id_counter = int(new_id) + 1

    item["instance_id"] = new_id
    item["image_id"] = f"{item['attribute']}_{new_id}"

    processed_data.append(item)

# 输出处理后的 JSON 到新文件
with open('dataset/face_attribute_dataset_mutilabel.json', 'w') as outfile:
    json.dump(processed_data, outfile, indent=4, ensure_ascii=False)

print("数据处理完成，输出文件：dataset/face_attribute_dataset_mutilabel.json")
