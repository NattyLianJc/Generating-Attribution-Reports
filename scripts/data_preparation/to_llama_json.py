import json
import os


def transform_record(record, question, dataset_prefix="dataset"):
    """
    将单个记录转换为新的格式：
      - messages: 第一个消息为 <image> 与 question 拼接（role 为 user），
                  第二个消息为 record 中的 caption（role 为 assistant）。
      - images: 只有一个元素，由 dataset_prefix 与 record 中的 img_path 拼接而成。
    """
    # 获取 img_path 和 caption，如果不存在则使用空字符串
    img_path = record.get("img_path", "")
    caption = record.get("caption", "")

    # 拼接 dataset 目录和 img_path，注意 os.path.join 会自动处理斜杠
    full_img_path = os.path.join(dataset_prefix, img_path)

    new_record = {
        "messages": [
            {
                "content": f"<image>{question}",
                "role": "user"
            },
            {
                "content": caption,
                "role": "assistant"
            }
        ],
        "images": [
            full_img_path
        ]
    }
    return new_record


def transform_json_files_list(json_file_list, output_path, question, dataset_prefix="dataset"):
    """
    接收一个 JSON 文件路径列表 json_file_list，
    读取每个 JSON 文件并将其中的记录转换为新格式，
    最后将所有记录合并写入到 output_path 指定的输出 JSON 文件中。
    """
    merged_records = []

    for json_file in json_file_list:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                # 尝试加载 JSON 数据，确保其格式正确
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file {json_file}: {e}")
            continue
        except Exception as e:
            print(f"Error reading file {json_file}: {e}")
            continue

        # 如果一个 JSON 文件的根元素不是列表，将其包装为列表
        if not isinstance(data, list):
            data = [data]

        for record in data:
            try:
                new_record = transform_record(record, question, dataset_prefix)
                merged_records.append(new_record)
            except Exception as e:
                print(f"Error processing record from file {json_file}: {e}")
                continue

    # 将合并后的所有记录写入输出文件
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_records, f, indent=4, ensure_ascii=False)
        print(f"转换完成，结果已保存到 {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"Error writing output file {output_path}: {e}")


if __name__ == "__main__":
    # JSON 文件的路径列表（请根据实际情况进行调整）
    json_file_list = [
        "dataset/train-mutilabel.json",
        "dataset/eval-mutilabel.json",
        "dataset/test-mutilabel.json"
    ]

    # 指定合并后的输出 JSON 文件路径
    output_json = "mmtt_dataset.json"

    # 新的问题文本，不涉及多边形表示，只要求详细描述修改内容
    question = (
        "Please analyze this AI-altered face and provide a detailed descriptive explanation "
        "of the modifications observed. For example, comment on skin texture, facial symmetry, "
        "and other noticeable changes. Please ensure your answer is comprehensive."
    )

    transform_json_files_list(json_file_list, output_json, question)
