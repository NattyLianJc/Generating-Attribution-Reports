import json
import os
from PIL import Image
from PIL import UnidentifiedImageError  # 用于捕获无效图像文件的错误
from tqdm import tqdm  # <--- 导入 tqdm


def analyze_and_categorize_records(original_json_path,
                                   top_level_path_for_masks,
                                   output_directory,
                                   mask_definition_is_nonzero=True,
                                   specific_mask_value=255):
    """
    分析JSON记录中的mask图像，根据mask占图像总面积的比例对记录进行分类，
    并将分类后的记录保存到新的JSON文件中，带有进度条。
    """
    try:
        with open(original_json_path, 'r', encoding='utf-8') as f:
            all_records = json.load(f)
    except FileNotFoundError:
        print(f"错误: 原始JSON文件未找到: {original_json_path}")
        return
    except json.JSONDecodeError:
        print(f"错误: 原始JSON文件格式无效: {original_json_path}")
        return

    if not isinstance(all_records, list):
        print(f"错误: 原始JSON文件的顶层结构应为一个列表 (list of records)。当前类型: {type(all_records)}")
        return

    if not os.path.exists(output_directory):
        try:
            os.makedirs(output_directory)
            print(f"已创建输出目录: {output_directory}")
        except OSError as e:
            print(f"错误: 无法创建输出目录 '{output_directory}': {e}")
            return

    categorized_records = [[] for _ in range(10)]
    level_bounds = [(i * 10, (i + 1) * 10) for i in range(10)]

    record_count_total = len(all_records)  # 获取记录总数以用于tqdm
    processed_count = 0
    skipped_due_missing_path = 0
    skipped_due_bad_image_or_size = 0
    error_count = 0

    # 使用 tqdm 包装 all_records 来显示进度条
    # desc 参数设置进度条的描述文字
    # unit 参数设置每个处理单元的名称
    print(f"\n开始处理 {record_count_total} 条记录...")
    for record_index, record in enumerate(tqdm(all_records, desc="处理进度", unit="条记录", ncols=100)):
        if not isinstance(record, dict):
            tqdm.write(f"警告: 在索引 {record_index} 处的记录不是一个字典对象，已跳过。记录内容: {record}")
            error_count += 1
            continue

        mask_relative_path = record.get("mask_path")
        image_id = record.get("image_id", f"记录_{record_index + 1}")

        if not mask_relative_path:
            # tqdm.write 会将输出打印在进度条上方，避免干扰
            tqdm.write(f"警告: 记录 '{image_id}' 缺少 'mask_path' 字段，已跳过。")
            skipped_due_missing_path += 1
            continue

        full_mask_path = os.path.normpath(os.path.join(top_level_path_for_masks, mask_relative_path))

        try:
            mask_image = Image.open(full_mask_path)
            mask_image_gray = mask_image.convert("L")
            width, height = mask_image_gray.size

            if width == 0 or height == 0:
                tqdm.write(f"警告: 记录 '{image_id}' 的 Mask 图像 '{full_mask_path}' 尺寸为0，已跳过。")
                skipped_due_bad_image_or_size += 1
                continue

            total_pixels = width * height
            mask_pixel_count = 0

            for pixel_value in mask_image_gray.getdata():
                if mask_definition_is_nonzero:
                    if pixel_value > 0:
                        mask_pixel_count += 1
                else:
                    if pixel_value == specific_mask_value:
                        mask_pixel_count += 1

            percentage = (mask_pixel_count / total_pixels) * 100
            processed_count += 1

            assigned_level = -1
            if percentage >= 100.0:
                assigned_level = 9
            elif percentage >= 90.0:
                assigned_level = 9
            else:
                assigned_level = int(percentage // 10)

            if 0 <= assigned_level < 10:
                categorized_records[assigned_level].append(record)
            else:
                tqdm.write(f"警告: 记录 '{image_id}' 计算出的比例 {percentage}% 无法分配到有效等级。将放入0-10%等级。")
                categorized_records[0].append(record)  # 默认放入第一个等级
                error_count += 1


        except FileNotFoundError:
            tqdm.write(f"错误 (文件未找到): 记录 '{image_id}' 的 Mask: '{full_mask_path}'")
            error_count += 1
        except UnidentifiedImageError:
            tqdm.write(
                f"错误 (无效图像): 记录 '{image_id}' 的 Mask 文件 '{full_mask_path}' 不是有效的图像文件或格式无法识别。")
            skipped_due_bad_image_or_size += 1
        except Exception as e:
            tqdm.write(f"错误 (处理中): 记录 '{image_id}' (Mask: '{full_mask_path}') 时发生意外错误: {e}")
            error_count += 1

    # 进度条完成后，打印总结信息
    print("\n--- 处理结果总结 ---")
    print(f"总共扫描记录: {record_count_total}条")
    print(f"成功处理并计算比例的记录: {processed_count}条")
    print(f"因缺少mask_path跳过的记录: {skipped_due_missing_path}条")
    print(f"因图像无效或尺寸为0跳过的记录: {skipped_due_bad_image_or_size}条")
    print(f"处理过程中发生其他错误的记录: {error_count}条")
    print("-----------------------\n")

    original_base_name = os.path.splitext(os.path.basename(original_json_path))[0]

    print("开始生成分类JSON文件...")
    for i in range(10):
        min_percent, max_percent_exclusive = level_bounds[i]
        if i == 9:
            level_name_suffix = f"{min_percent}_100_percent"
        else:
            level_name_suffix = f"{min_percent}_{max_percent_exclusive}_percent"

        new_json_filename = f"{original_base_name}_{level_name_suffix}.json"
        output_path = os.path.join(output_directory, new_json_filename)

        if not categorized_records[i]:
            print(f"信息: 等级 {level_name_suffix.replace('_percent', '')} 没有记录，不生成文件 {output_path}。")
            continue

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(categorized_records[i], f, ensure_ascii=False, indent=4)
            print(f"成功: 已生成文件 '{output_path}' 包含 {len(categorized_records[i])} 条记录。")
        except Exception as e:
            print(f"错误: 写入新的JSON文件 '{output_path}' 失败: {e}")


# --- 主程序入口 ---
if __name__ == '__main__':
    # ---- 用户需要配置以下参数 ----

    # 1. 原始JSON文件的完整路径
    input_json_file = "dataset/eval-mutilabel.json"  # <--- 【请修改这里】

    # 2. 顶层路径 (用于解析 mask_path)
    top_path_for_mask_files = "dataset"  # <--- 【请修改这里】

    # 3. 输出目录 (用于存放新生成的10个JSON文件)
    output_dir = "classified_json_output"  # <--- 【请修改这里】

    # 4. Mask 定义
    define_mask_as_nonzero_pixels = False  # <--- 【请根据你的mask图像格式修改】

    # 5. 如果 define_mask_as_nonzero_pixels 为 False, 指定mask的像素值
    specific_value_for_mask_pixel = 255  # <--- 【如果需要，请修改这里】

    # ---- 参数配置结束 ----

    print("脚本开始执行...")
    if input_json_file == "YOUR_ORIGINAL_JSON_FILE.json" or not os.path.exists(input_json_file):
        print("-" * 50)
        print("错误：请务必修改脚本中的 'input_json_file' 参数以指向你的原始JSON文件。")
        print(f"当前设置的路径 '{input_json_file}' 不存在或仍为默认值。")
        print("脚本将不会运行。")
        print("-" * 50)
    else:
        print(f"开始处理文件: {input_json_file}")
        print(
            f"顶层路径 (用于mask文件) 设置为: '{top_path_for_mask_files if top_path_for_mask_files else '将直接使用mask_path (假设为相对或绝对路径)'}'")
        print(f"输出目录设置为: {os.path.abspath(output_dir)}")
        print(
            f"Mask 定义: {'非零像素' if define_mask_as_nonzero_pixels else f'像素值为 {specific_value_for_mask_pixel}'}")

        analyze_and_categorize_records(
            original_json_path=input_json_file,
            top_level_path_for_masks=top_path_for_mask_files,
            output_directory=output_dir,
            mask_definition_is_nonzero=define_mask_as_nonzero_pixels,
            specific_mask_value=specific_value_for_mask_pixel
        )
        print("\n处理完成。")