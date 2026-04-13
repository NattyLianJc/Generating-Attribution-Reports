import json
import cv2
import os
import shutil
import copy
import argparse
import concurrent.futures
from tqdm import tqdm

# --- 1. 定义多进程的 Worker 函数 ---
def process_single_record(item, input_base_dir, output_base_dir, scale, kernel_size, replace_prefix, dir_configs):
    """
    处理单张图片的逻辑
    """
    # 强制单进程单线程，防止 OpenCV 和 multiprocessing 调度冲突
    cv2.setNumThreads(1) 

    old_img_path = item["img_path"]    
    old_mask_path = item["mask_path"]  
    
    # 输入的物理路径
    real_read_img_path = os.path.join(input_base_dir, old_img_path)
    real_read_mask_path = os.path.join(input_base_dir, old_mask_path)
    
    # 1. 读取原始图像
    image_bgr = cv2.imread(real_read_img_path)
    if image_bgr is None:
        return {"error": f"跳过：无法读取 {real_read_img_path}"}

    # 2. 定义内部退化函数
    def apply_downscale(img):
        h, w = img.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        low_res = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return cv2.resize(low_res, (w, h), interpolation=cv2.INTER_NEAREST)

    def apply_blur(img):
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    # 3. 映射对应的配置 (名字和处理好的图像)
    variants = {
        "downscale": {"dir_name": dir_configs["downscale"], "img": apply_downscale(image_bgr)},
        "blur":      {"dir_name": dir_configs["blur"],      "img": apply_blur(image_bgr)},
        "both":      {"dir_name": dir_configs["both"],      "img": apply_blur(apply_downscale(image_bgr))}
    }

    # 4. 保存物理文件并构建新的 JSON 结构
    results = {}
    for key, variant_info in variants.items():
        variant_dir = variant_info["dir_name"]
        img_transformed = variant_info["img"]
        
        # A. 更新 JSON 里的路径字段 (替换前缀为新的顶级文件夹名)
        new_img_path_json = old_img_path.replace(replace_prefix, f"{variant_dir}/", 1)
        new_mask_path_json = old_mask_path.replace(replace_prefix, f"{variant_dir}/", 1)
        
        # B. 构造实际保存到硬盘的路径 (output_base_dir / new_path)
        real_save_img_path = os.path.join(output_base_dir, new_img_path_json)
        real_save_mask_path = os.path.join(output_base_dir, new_mask_path_json)
        
        # C. 创建目录
        os.makedirs(os.path.dirname(real_save_img_path), exist_ok=True)
        os.makedirs(os.path.dirname(real_save_mask_path), exist_ok=True)
        
        # D. 保存图像
        cv2.imwrite(real_save_img_path, img_transformed)
        
        # E. 物理复制 Mask
        if os.path.exists(real_read_mask_path):
            shutil.copy2(real_read_mask_path, real_save_mask_path)
        
        # F. 组装新记录
        new_item = copy.deepcopy(item)
        new_item["img_path"] = new_img_path_json
        new_item["mask_path"] = new_mask_path_json
        results[key] = new_item

    return {"success": results}


def main():
    # --- 2. 命令行参数设置 ---
    parser = argparse.ArgumentParser(description="Fully Parameterized High-Performance Image Degradation Pipeline")
    
    # 【输入输出路径设置】
    parser.add_argument("--input_base_dir", type=str, default="dataset", help="输入的基础数据集目录")
    parser.add_argument("--input_json_name", type=str, default="eval-mutilabel.json", help="输入的 JSON 文件名")
    parser.add_argument("--output_base_dir", type=str, default="dataset", help="输出的基础数据集目录")
    parser.add_argument("--replace_prefix", type=str, default="dataset/", help="JSON 路径中需要被替换掉的前缀字符串")
    
    # 【生成的顶级文件夹名称设置】
    parser.add_argument("--dir_downscale", type=str, default="dataset_downscale", help="下采样图片保存的顶级文件夹名")
    parser.add_argument("--dir_blur", type=str, default="dataset_blur", help="模糊图片保存的顶级文件夹名")
    parser.add_argument("--dir_both", type=str, default="dataset_both", help="双重退化图片保存的顶级文件夹名")
    
    # 【生成的 JSON 文件名称设置】
    parser.add_argument("--json_downscale", type=str, default="eval-mutilabel_downscale.json", help="下采样结果 JSON 文件名")
    parser.add_argument("--json_blur", type=str, default="eval-mutilabel_blur.json", help="模糊结果 JSON 文件名")
    parser.add_argument("--json_both", type=str, default="eval-mutilabel_both.json", help="双重退化结果 JSON 文件名")

    # 【算法与性能参数】
    parser.add_argument("--scale", type=float, default=0.75, help="下采样比例 (默认 0.75)")
    parser.add_argument("--kernel_size", type=int, default=5, help="高斯模糊核大小 (必须为奇数，默认 5)")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="使用的 CPU 进程数 (默认使用全部核心)")
    parser.add_argument("--limit", type=int, default=None, help="限制处理的图片数量 (用于快速测试)")
    
    args = parser.parse_args()

    json_path = os.path.join(args.input_base_dir, args.input_json_name)

    if not os.path.exists(json_path):
        print(f"❌ 错误: 找不到输入 JSON 文件 {json_path}")
        return

    # --- 3. 读取数据 ---
    print(f"📖 读取 JSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        original_dataset = json.load(f)

    if args.limit:
        original_dataset = original_dataset[:args.limit]
        print(f"⚠️ Limit 设置生效，只处理前 {args.limit} 条记录。")

    dataset_downscale, dataset_blur, dataset_both = [], [], []
    
    # 打包目录配置传给 worker
    dir_configs = {
        "downscale": args.dir_downscale,
        "blur": args.dir_blur,
        "both": args.dir_both
    }

    print(f"🚀 开始并行处理，总计 {len(original_dataset)} 条记录...")
    print(f"⚙️  进程数: {args.num_workers} | 输出根目录: {args.output_base_dir}")

    # 自动创建输出根目录
    os.makedirs(args.output_base_dir, exist_ok=True)

    # --- 4. 开启多进程池 ---
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                process_single_record, 
                item, 
                args.input_base_dir, 
                args.output_base_dir,
                args.scale, 
                args.kernel_size,
                args.replace_prefix,
                dir_configs
            ): item for item in original_dataset
        }

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(original_dataset), desc="Processing Images"):
            result = future.result()
            
            if "error" in result:
                print(result["error"])
            elif "success" in result:
                data = result["success"]
                dataset_downscale.append(data["downscale"])
                dataset_blur.append(data["blur"])
                dataset_both.append(data["both"])

    # --- 5. 保存三个新的 JSON 文件 ---
    print("\n💾 正在保存生成的 JSON 文件...")
    output_configs = [
        (args.json_downscale, dataset_downscale),
        (args.json_blur,      dataset_blur),
        (args.json_both,      dataset_both)
    ]

    for filename, data_list in output_configs:
        save_path = os.path.join(args.output_base_dir, filename)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=4, ensure_ascii=False)
        print(f"✅ 成功保存: {save_path}")

    print("\n🎉 所有任务已完成！结构如下：")
    print(f"|- {args.output_base_dir}/")
    print(f"   |- {args.dir_downscale}/ ")
    print(f"   |- {args.dir_blur}/ ")
    print(f"   |- {args.dir_both}/ ")
    print(f"   |- {args.json_downscale}")
    print(f"   |- {args.json_blur}")
    print(f"   |- {args.json_both}")

if __name__ == "__main__":
    main()