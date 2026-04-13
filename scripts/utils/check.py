import json
import os

def check_json_structure(file_path):
    try:
        with open(file_path, 'r') as f:
            dataset = json.load(f)
            if isinstance(dataset, dict):
                print(f"The file {file_path} is a valid JSON dictionary.")
            else:
                print(f"Error: The file {file_path} is not a valid JSON dictionary. It is a {type(dataset)}.")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON file {file_path}. Error: {str(e)}")
    except Exception as e:
        print(f"Error: An error occurred while reading the file {file_path}. Error: {str(e)}")

# 指定要检查的 JSON 文件路径
annotation_file = '/home/share/qrn8y2ug/home/ysuanAJ27/ljc/LAVIS-main/temp/                                                                                                                                                                                                                     test_gt.json'

# 检查 JSON 文件结构
check_json_structure(annotation_file)
