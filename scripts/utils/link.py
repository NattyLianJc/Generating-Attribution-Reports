import os
import pathlib
import json # 导入 json 模块

def create_symlinks_for_dataset(dataset_list, top_level_dir, output_base_dir):
    """
    为数据集中的图片和掩码创建软链接。

    参数:
    dataset_list (list): 包含数据记录的列表。
                         每个记录是一个字典，至少包含 "img_path", "mask_path", 和 "image_id"。
    top_level_dir (str): 包含原始数据集的顶层目录路径。
                         "img_path" 和 "mask_path" 是相对于此目录的路径。
    output_base_dir (str): 将在其中创建 "img" 和 "mask" 子文件夹并存放软链接的目录路径。
    """
    img_symlink_dir = os.path.join(output_base_dir, "img")
    mask_symlink_dir = os.path.join(output_base_dir, "mask")

    os.makedirs(img_symlink_dir, exist_ok=True)
    os.makedirs(mask_symlink_dir, exist_ok=True)

    print(f"将在以下位置创建软链接：")
    print(f"图片软链接目录: {img_symlink_dir}")
    print(f"掩码软链接目录: {mask_symlink_dir}")
    print("-" * 30)

    processed_count = 0
    error_count = 0

    for item in dataset_list:
        original_img_full_path_str = "未知" # 用于错误报告
        symlink_img_target_path_str = "未知" # 用于错误报告
        try:
            image_id = item["image_id"]
            original_img_relative_path = item["img_path"]
            original_mask_relative_path = item["mask_path"]

            original_img_full_path = pathlib.Path(top_level_dir) / original_img_relative_path
            original_mask_full_path = pathlib.Path(top_level_dir) / original_mask_relative_path
            original_img_full_path_str = str(original_img_full_path) # 更新用于错误报告的变量

            if not original_img_full_path.exists():
                print(f"错误: 原始图片文件不存在: {original_img_full_path} (image_id: {image_id})")
                error_count += 1
                continue
            if not original_mask_full_path.exists():
                print(f"错误: 原始掩码文件不存在: {original_mask_full_path} (image_id: {image_id})")
                error_count += 1
                continue

            img_extension = original_img_full_path.suffix
            mask_extension = original_mask_full_path.suffix

            new_img_filename = f"{image_id}{img_extension}"
            new_mask_filename = f"{image_id}{mask_extension}"

            symlink_img_target_path = os.path.join(img_symlink_dir, new_img_filename)
            symlink_mask_target_path = os.path.join(mask_symlink_dir, new_mask_filename)
            symlink_img_target_path_str = symlink_img_target_path # 更新用于错误报告的变量

            if not os.path.exists(symlink_img_target_path) and not os.path.islink(symlink_img_target_path):
                os.symlink(original_img_full_path.resolve(), symlink_img_target_path)
            else:
                print(f"  跳过: 图片软链接已存在 {symlink_img_target_path}")

            if not os.path.exists(symlink_mask_target_path) and not os.path.islink(symlink_mask_target_path):
                os.symlink(original_mask_full_path.resolve(), symlink_mask_target_path)
            else:
                print(f"  跳过: 掩码软链接已存在 {symlink_mask_target_path}")

            processed_count += 1

        except KeyError as e:
            print(f"错误: 数据记录缺少键: {e}。记录内容: {item}")
            error_count += 1
        except OSError as e:
            print(f"错误: 创建软链接时发生 OS 错误 (image_id: {item.get('image_id', '未知')}, 文件: {original_img_relative_path if 'original_img_relative_path' in locals() else '未知'}): {e}")
            print(f"  请检查是否有足够的权限，或者源文件路径是否正确。")
            print(f"  原始图片路径: {original_img_full_path_str}")
            print(f"  软链接目标路径: {symlink_img_target_path_str}")
            error_count += 1
        except Exception as e:
            print(f"错误: 处理记录时发生未知错误 (image_id: {item.get('image_id', '未知')}): {e}")
            error_count += 1

    print("-" * 30)
    print(f"处理完成。")
    print(f"成功处理并创建/检查软链接的记录数: {processed_count}")
    print(f"发生错误的记录数: {error_count}")

# --- 如何使用这个脚本 ---

if __name__ == "__main__":
    # 1. 指定你的 JSON 数据文件路径
    # ---- 请修改以下路径为你实际的 JSON 文件路径 ----
    your_json_file_path = "path/to/your/dataset_file.json" # <--- 修改这里

    # 2. 指定你的顶层目录 (包含 JSON 中相对路径所指向的文件的根目录)
    # ---- 请修改以下路径为你实际的路径 ----
    your_top_level_directory = "/home/user/my_project/main_dataset_folder" # <--- 修改这里

    # 3. 指定输出软链接的目录
    # ---- 请修改以下路径为你实际的路径 ----
    your_output_directory = "/home/user/my_project/organized_symlinks" # <--- 修改这里

    # 从 JSON 文件加载数据
    try:
        with open(your_json_file_path, 'r', encoding='utf-8') as f:
            dataset_list_from_json = json.load(f)
        print(f"成功从 {your_json_file_path} 加载了 {len(dataset_list_from_json)} 条数据记录。")
    except FileNotFoundError:
        print(f"错误: JSON 文件未找到: {your_json_file_path}")
        exit(1) # 退出脚本
    except json.JSONDecodeError:
        print(f"错误: JSON 文件格式无效或解析失败: {your_json_file_path}")
        exit(1) # 退出脚本
    except Exception as e:
        print(f"加载 JSON 文件时发生未知错误: {e}")
        exit(1) # 退出脚本

    # 确保加载的数据是一个列表 (你的JSON文件顶层应该是一个列表)
    if not isinstance(dataset_list_from_json, list):
        print(f"错误: 从 JSON 文件加载的数据不是一个列表。请检查 '{your_json_file_path}' 的内容。")
        exit(1)

    # --- (可选) 用于测试的虚拟文件创建 ---
    # def setup_dummy_files_for_testing(top_dir, data):
    #     print("\n--- 设置测试用的虚拟文件和目录 ---")
    #     for item in data:
    #         try:
    #             img_file = pathlib.Path(top_dir) / item["img_path"]
    #             mask_file = pathlib.Path(top_dir) / item["mask_path"]
    #             img_file.parent.mkdir(parents=True, exist_ok=True)
    #             mask_file.parent.mkdir(parents=True, exist_ok=True)
    #             if not img_file.exists(): img_file.touch()
    #             if not mask_file.exists(): mask_file.touch()
    #         except Exception as e: print(f"  创建虚拟文件失败 {item.get('image_id', '')}: {e}")
    #     print("--- 虚拟文件设置完毕 ---\n")
    #
    # # 如果需要测试，取消下面两行的注释，并确保路径正确
    # # print("注意：如果启用了 setup_dummy_files_for_testing，请确保 your_top_level_directory 已创建。")
    # # setup_dummy_files_for_testing(your_top_level_directory, dataset_list_from_json)

    # 调用核心函数处理数据
    create_symlinks_for_dataset(dataset_list_from_json, your_top_level_directory, your_output_directory)