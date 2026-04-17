import json
import os


def process_json_file(input_filepath):
    """
    处理单个JSON文件：
    1. 删除 image_id 以 "5_" 开头的条目。
    2. 删除剩余条目中 caption 的第一句话。
    3. 将结果保存到新的JSON文件中。

    参数:
    input_filepath (str): 输入的JSON文件路径。
    """
    # 检查文件是否存在
    if not os.path.exists(input_filepath):
        print(f"错误：文件 '{input_filepath}' 不存在。")
        return

    # 定义输出文件路径
    directory, filename = os.path.split(input_filepath)
    name, ext = os.path.splitext(filename)
    output_filepath = os.path.join(directory, f"{name}_processed{ext}")

    try:
        # 读取原始JSON文件
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        processed_data = []

        # 遍历每一条记录
        for record in data:
            # 检查 image_id 是否符合要求 (不存在以 "5_" 开头的记录)
            if 'image_id' in record and not str(record.get('image_id', '')).startswith('5_'):

                # # 处理 caption
                # if 'caption' in record and isinstance(record['caption'], str):
                #     caption = record['caption']
                #     # 寻找第一个句号的位置
                #     first_period_index = caption.find('.')
                #
                #     # 如果找到了句号
                #     if first_period_index != -1:
                #         # 截取第一个句号之后的内容，并移除可能存在的前导空格
                #         new_caption = caption[first_period_index + 1:].lstrip()
                #         record['caption'] = new_caption
                #     # 如果没有找到句号，caption 保持不变

                # 将处理后的记录添加到新列表中
                processed_data.append(record)

        # 将处理后的数据写入新文件
        with open(output_filepath, 'w', encoding='utf-8') as f:
            # ensure_ascii=False 保证中文字符正确显示
            # indent=4         使JSON文件格式化，易于阅读
            json.dump(processed_data, f, ensure_ascii=False, indent=4)

        print(f"处理完成！\n输入文件: {input_filepath}\n输出文件: {output_filepath}")

    except json.JSONDecodeError:
        print(f"错误：文件 '{input_filepath}' 不是一个有效的JSON格式。")
    except Exception as e:
        print(f"处理文件 '{input_filepath}' 时发生未知错误: {e}")


# --- 使用说明 ---
if __name__ == "__main__":
    # 1. 将你的两个JSON文件名放入这个列表中
    files_to_process = ["dataset/test_df_cut.json", "dataset/train_df_cut.json"]

    # 2. 确保这个Python脚本和你的JSON文件在同一个文件夹下
    #    或者在下面的文件名中使用绝对路径

    # 3. 运行这个脚本
    for file in files_to_process:
        process_json_file(file)