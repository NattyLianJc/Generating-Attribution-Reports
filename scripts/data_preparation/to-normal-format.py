import json
import os

# 根目录路径
ROOT_DIR = "/home/share/qrn8y2ug/home/ysuanAJ27/ljc/LAVIS-main/dataset"

# 检查路径是否存在
def check_path_exists(path):
    full_path = os.path.join(ROOT_DIR, path)
    return os.path.exists(full_path)

# 读取路径JSON文件
with open('/home/share/qrn8y2ug/home/ysuanAJ27/ljc/LAVIS-main/dataset/paths.json', 'r') as paths_file:
    paths_data = json.load(paths_file)

# 读取caption JSON文件
with open('/home/share/qrn8y2ug/home/ysuanAJ27/ljc/LAVIS-main/handle_caption/output.json', 'r') as captions_file:
    captions_data = json.load(captions_file)

# 生成新的JSON数据
new_data = []
errors = []
for image_id, caption in captions_data.items():
    if image_id in paths_data:
        img_path = paths_data[image_id]['image']
        mask_path = paths_data[image_id]['mask']
        img_full_path = os.path.join(ROOT_DIR, img_path)
        mask_full_path = os.path.join(ROOT_DIR, mask_path)
        if not check_path_exists(img_path):
            errors.append(f"Image path error for {image_id}: {img_full_path} does not exist.")
        if not check_path_exists(mask_path):
            errors.append(f"Mask path error for {image_id}: {mask_full_path} does not exist.")
        new_data.append({
            "caption": caption,
            "image_id": image_id,
            "img_path": img_path,
            "mask_path": mask_path
        })
    else:
        errors.append(f"Path not found for image_id: {image_id}")

# 写入输出JSON文件
with open('/home/share/qrn8y2ug/home/ysuanAJ27/ljc/LAVIS-main/handle_caption/test/output-normal.json', 'w') as outfile:
    json.dump(new_data, outfile, indent=4)

# 输出错误信息
if errors:
    for error in errors:
        print(error)
else:
    print("New JSON file has been created successfully.")
