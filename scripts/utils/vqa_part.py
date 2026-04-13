import json
import os
from PIL import Image
from lavis.models import load_model_and_preprocess
import torch

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和预处理器
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="aokvqa", is_eval=True, device=device)

# 设置根目录
root_dir = "/home/share/qrn8y2ug/home/ysuanAJ27/ljc/LAVIS-main/dataset/"

# 加载输入JSON文件
with open("/home/share/qrn8y2ug/home/ysuanAJ27/ljc/LAVIS-main/dataset/train.json", "r") as f:
    data = json.load(f)

# 要处理的前几个记录数量
num_records_to_process = 100

# 定义同义词字典
facial_features = {
    "eye": ["eye", "eyes", "ocular", "optic"],
    "pupil": ["pupil", "pupils"],
    "lip": ["lip", "lips", "labium", "labia"],
    "mouth": ["mouth", "oral cavity", "buccal"],
    "nose": ["nose", "nasal", "nostril", "nostrils"],
    "ear": ["ear", "ears", "auricle", "pinna"],
    "cheek": ["cheek", "cheeks", "buccal", "malar"],
    "forehead": ["forehead", "frontal", "brow"],
    "chin": ["chin", "mentum"],
    "jaw": ["jaw", "mandible", "maxilla"],
    "eyebrow": ["eyebrow", "eyebrows"],
    "tooth": ["tooth", "teeth", "dentition"],
    "tongue": ["tongue", "glossa", "lingua"],
    "beard": ["beard", "facial hair"],
    "mustache": ["mustache", "moustache"],
    "skin": ["skin", "derma", "epidermis"],
    "hair": ["hair", "locks", "tresses"],
    "wrinkle": ["wrinkle", "wrinkles", "line", "lines"],
    "freckle": ["freckle", "freckles", "lentigo", "lentigines"],
    "scar": ["scar", "scars", "cicatrix", "cicatrices"],
    "dimple": ["dimple", "dimples"]
}

# 创建逆向字典，用于快速查找同义词对应的主要词
reverse_facial_features = {}
for key, synonyms in facial_features.items():
    for synonym in synonyms:
        reverse_facial_features[synonym] = key

def extract_keywords(text):
    extracted = set()
    for word in text.lower().split():
        cleaned_word = word.strip(",.!?;:()[]{}")
        if cleaned_word in reverse_facial_features:
            extracted.add(reverse_facial_features[cleaned_word])
    return list(extracted)

def calculate_overlap(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    overlap = set1.intersection(set2)
    return len(overlap) / len(set1.union(set2))

results = []
total_overlap_score = 0

for record in data[:num_records_to_process]:
    image_id = record["image_id"]
    img_path = os.path.join(root_dir, record["img_path"])
    caption = record["caption"]

    # 加载并处理图像
    raw_image = Image.open(img_path).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    # question = "This face is synthesized by AI. Please list all parts of this face that are synthesized by AI, such as the beard, clothes, etc."
    question = (
        "This face is synthesized by AI. Please list all parts list all parts you suspect to be synthesized, such as the hair, beard, clothes, etc."
    )

    question = txt_processors["eval"](question)

    # 获取模型的预测答案
    # answers = model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")
    answers = model.predict_answers(
        samples={"image": image, "text_input": question},
        inference_method="generate",
        top_k=50,         # 设置top_k
        top_p=0.9,        # 设置top_p
        temperature=1.2,  # 设置temperature
        max_length=60     # 可以根据需要调整
        # num_beams=5       # 可以根据需要调整
    )

    generated_keywords = extract_keywords(answers[0]) if answers else []

    # 从caption中提取关键词
    caption_keywords = extract_keywords(caption)

    # 计算重合度
    overlap_score = calculate_overlap(generated_keywords, caption_keywords)
    total_overlap_score += overlap_score

    # 保存结果
    result = {
        "image_id": image_id,
        "generated_keywords": generated_keywords,
        "caption_keywords": caption_keywords,
        "overlap_score": overlap_score
    }
    results.append(result)

average_overlap_score = total_overlap_score / num_records_to_process
print(f"Average overlap score: {average_overlap_score}")

# 将结果写入输出JSON文件
with open("output.json", "w") as f:
    json.dump(results, f, indent=4)

print("Processing complete. Results saved to output.json")
