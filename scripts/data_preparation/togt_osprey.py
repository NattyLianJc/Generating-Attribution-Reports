import json
import glob
import os

# 指定输入和输出文件夹
input_folder = 'temp'
output_folder = 'temp'
input_files = [f'{input_folder}/train-mutilabel.json', f'{input_folder}/eval-mutilabel.json', f'{input_folder}/test-mutilabel.json']

# 处理文件并生成目标格式
for input_file in input_files:
    with open(input_file, 'r') as f:
        data = json.load(f)
        new_data = {'annotations': [], 'images': []}
        image_ids = set()
        for item in data:
            if 'instance_id' in item:
                new_data['annotations'].append({
                    'image_id': item['image_id'],
                    'caption': item['caption'],
                    'id': item['instance_id'],
                    'img_path': item["img_path"],
                    'mask_path': item["mask_path"]
                })
                if item['image_id'] not in image_ids:
                    new_data['images'].append({
                        'id': item['image_id']
                        # 'file_name': item['img_path']  # 你需要根据实际情况调整这个字段
                    })
                    image_ids.add(item['image_id'])
            else:
                print(f"Error: instance_id not found in item: {item}")

    # 保存更新后的 JSON 文件
    output_file = os.path.join(output_folder, os.path.basename(input_file).replace(".json", "_gt.json"))
    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=4)

    print(f"Updated JSON file saved to {output_file}")

print("All files have been processed and saved.")

