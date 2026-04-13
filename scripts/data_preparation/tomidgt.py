import json
import re
import glob
from collections import defaultdict

# 指定输入和输出文件夹
input_folder = '/home/share/qrn8y2ug/home/ysuanAJ27/ljc/LAVIS-main/midtemp'
output_folder = '/home/share/qrn8y2ug/home/ysuanAJ27/ljc/LAVIS-main/temp'
input_files = [f'{input_folder}/train.json', f'{input_folder}/test.json', f'{input_folder}/eval.json']

prefix_count = defaultdict(int)

# 第一遍读取所有文件，统计前缀
all_data = []
for input_file in input_files:
    with open(input_file, 'r') as f:
        data = json.load(f)
        all_data.append((input_file, data))
        for item in data:
            match = re.match(r'([a-zA-Z0-9]+)_', item['image_id'])
            if match:
                prefix = match.group(1)
                prefix_count[prefix] += 1
            else:
                print(f"Warning: Unable to match prefix in image_id '{item['image_id']}'")

# 为每个前缀分配唯一的数字
prefix_to_number = {prefix: idx + 1 for idx, prefix in enumerate(prefix_count.keys())}

# 打印前缀映射以进行调试
print("Prefix to number mapping:")
for prefix, number in prefix_to_number.items():
    print(f"{prefix}: {number}")

# 第二遍读取所有文件，更新 `instance_id`
for input_file, data in all_data:
    for item in data:
        match = re.match(r'([a-zA-Z0-9]+)_(\d+)', item['image_id'])
        if match:
            prefix = match.group(1)
            number_part = match.group(2)
            if prefix in prefix_to_number:
                instance_id = f"{prefix_to_number[prefix]}{number_part}"
                item['instance_id'] = instance_id
            else:
                print(f"Error: Prefix '{prefix}' not found in prefix_to_number mapping.")
        else:
            print(f"Error: Invalid image_id format '{item['image_id']}'")

    # 保存更新后的 JSON 文件
    output_file = f'{output_folder}/{input_file.split("/")[-1]}'
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Updated JSON file saved to {output_file}")

print("All files have been processed and saved.")