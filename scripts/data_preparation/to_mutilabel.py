import json
import re

# # 定义同义词字典
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

# # 定义同义词字典
# facial_features = {
#     "eye_area": ["eye", "eyes", "ocular", "optic", "pupil", "pupils", "eyebrow", "eyebrows"],
#     "mouth_area": ["lip", "lips", "labium", "labia", "mouth", "oral cavity", "buccal", "tooth", "teeth", "dentition", "tongue", "glossa", "lingua"],
#     "nose_area": ["nose", "nasal", "nostril", "nostrils"],
#     "ear_area": ["ear", "ears", "auricle", "pinna"],
#     "cheek_area": ["cheek", "cheeks", "buccal", "malar", "chin", "mentum", "jaw", "mandible", "maxilla"],
#     "forehead_area": ["forehead", "frontal", "brow"],
#     "facial_hair": ["beard", "mustache", "moustache", "facial hair"],
#     "skin_conditions": ["skin", "derma", "epidermis", "wrinkle", "wrinkles", "line", "lines", "freckle", "freckles", "lentigo", "lentigines", "scar", "scars", "cicatrix", "cicatrices"],
#     "other_features": ["dimple", "dimples"]
# }

# 定义标签顺序
labels_order = list(facial_features.keys())

# 打印标签的顺序
print("标签顺序:", labels_order)

# 读取JSON文件
with open('dataset/processed_test.json', 'r') as file:
    data = json.load(file)

# 提取关键词的函数
def extract_keywords(caption):
    keywords = []
    for feature, synonyms in facial_features.items():
        for synonym in synonyms:
            if re.search(r'\b' + re.escape(synonym) + r'\b', caption, re.IGNORECASE):
                keywords.append(feature)
                break
    return keywords

# 转换关键词为多分类标签的函数
def keywords_to_multilabel(keywords):
    label_vector = [0] * len(labels_order)
    for keyword in keywords:
        if keyword in labels_order:
            index = labels_order.index(keyword)
            label_vector[index] = 1
    return label_vector

# 处理数据
processed_data = []
for item in data:
    keywords = extract_keywords(item["caption"])
    multilabel = keywords_to_multilabel(keywords)
    item["multilabel"] = multilabel
    processed_data.append(item)

# 输出新的JSON文件
with open('dataset/processed_test.json', 'w') as outfile:
    json.dump(processed_data, outfile, indent=4)
