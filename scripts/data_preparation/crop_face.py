import cv2
import numpy as np
import os
import json
from tqdm import tqdm


# ==============================================================================
# 核心功能函数：裁剪与缩放 (此部分无需修改)
# ==============================================================================
def crop_and_resize(image, crop_bbox, margin_scale=1.2, target_size=(512, 512)):
    """
    根据给定的边界框，对图像进行正方形裁剪，并缩放到指定大小。
    """
    x, y, w, h = crop_bbox
    center_x = x + w / 2
    center_y = y + h / 2
    side_len = max(w, h) * margin_scale

    square_x = center_x - side_len / 2
    square_y = center_y - side_len / 2

    top_pad = max(0, -int(square_y))
    bottom_pad = max(0, int(square_y + side_len) - image.shape[0])
    left_pad = max(0, -int(square_x))
    right_pad = max(0, int(square_x + side_len) - image.shape[1])

    padded_image = cv2.copyMakeBorder(
        image, top_pad, bottom_pad, left_pad, right_pad,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    final_x = int(square_x) + left_pad
    final_y = int(square_y) + top_pad
    final_side = int(side_len)

    cropped_image = padded_image[final_y: final_y + final_side, final_x: final_x + final_side]

    if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
        return None

    if cropped_image.shape[0] > target_size[0]:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC

    resized_image = cv2.resize(cropped_image, target_size, interpolation=interpolation)

    return resized_image


# ==============================================================================
# 主处理流程 (已更新)
# ==============================================================================
def process_json_dataset(input_json_path, output_json_path, data_root, net):
    """
    处理整个JSON数据集。
    【新功能】：如果mask文件不存在，则自动创建一个全黑的mask。
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_data = []

    for item in tqdm(data, desc=f"正在处理 {os.path.basename(input_json_path)}"):
        # 1. 从JSON获取相对路径
        img_rel_path = item["img_path"]
        mask_rel_path = item.get("mask_path", "")  # 使用.get避免mask_path不存在时报错

        # 2. 与数据根目录拼接，得到完整的读取路径
        img_full_read_path = os.path.join(data_root, img_rel_path)

        # ======================= 核心修改开始 =======================

        # 分别检查image和mask
        if not os.path.exists(img_full_read_path):
            print(f"\n警告：跳过记录 {item['image_id']}，找不到图像文件: {img_full_read_path}")
            continue

        image = cv2.imread(img_full_read_path)
        if image is None:
            print(f"\n警告：跳过记录 {item['image_id']}，无法读取图像文件。")
            continue

        # 检查mask是否存在
        mask = None
        if mask_rel_path:
            mask_full_read_path = os.path.join(data_root, mask_rel_path)
            if os.path.exists(mask_full_read_path):
                mask = cv2.imread(mask_full_read_path)

        # 此时，如果mask文件不存在或路径为空，mask变量的值就是None

        # ... [人脸检测部分与之前完全相同] ...
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        best_face_bbox = None
        highest_confidence = 0.0

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > highest_confidence and confidence > 0.5:
                highest_confidence = confidence
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                best_face_bbox = (startX, startY, endX - startX, endY - startY)

        if best_face_bbox is None:
            print(f"\n警告：在 {item['image_id']} 中未检测到人脸，跳过。")
            continue

        # 定义目标尺寸
        target_size = (512, 512)

        # 裁剪主图像
        cropped_image = crop_and_resize(image, best_face_bbox, target_size=target_size)

        # 根据mask是否存在，进行不同处理
        if mask is not None:
            # 如果mask存在，正常裁剪
            cropped_mask = crop_and_resize(mask, best_face_bbox, target_size=target_size)
        else:
            # 如果mask不存在(mask is None)，则创建一个全黑的mask
            print(f"\n信息：在 {item['image_id']} 中未找到mask，将生成一个全黑mask。")
            # 使用Numpy创建一个形状为(height, width, channels)的全0数组，即黑色图片
            cropped_mask = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        # ======================== 核心修改结束 ========================

        if cropped_image is None or cropped_mask is None:
            print(f"\n警告：在 {item['image_id']} 中裁剪失败，跳过。")
            continue

        # 3. 生成新的相对路径
        img_parts = img_rel_path.split(os.sep)
        img_parts[0] = f"{img_parts[0]}_cut"
        new_img_rel_path = os.path.join(*img_parts)

        # 即使原始mask路径为空，也为其生成一个合法的保存路径
        if mask_rel_path:
            mask_parts = mask_rel_path.split(os.sep)
        else:
            # 如果原始mask路径不存在，就基于image路径来创造一个
            mask_parts = img_rel_path.split(os.sep)
            # 例如: DQ_FF++/manipulated.../images -> DQ_FF++/manipulated.../masks
            mask_parts = [p.replace('images', 'masks') for p in mask_parts]

        mask_parts[0] = f"{mask_parts[0]}_cut"
        new_mask_rel_path = os.path.join(*mask_parts)

        # 4. 拼接得到完整的写入路径
        new_img_full_write_path = os.path.join(data_root, new_img_rel_path)
        new_mask_full_write_path = os.path.join(data_root, new_mask_rel_path)

        # 5. 保存文件
        os.makedirs(os.path.dirname(new_img_full_write_path), exist_ok=True)
        os.makedirs(os.path.dirname(new_mask_full_write_path), exist_ok=True)
        cv2.imwrite(new_img_full_write_path, cropped_image)
        cv2.imwrite(new_mask_full_write_path, cropped_mask)  # 写入真实mask或黑色mask

        # 6. 更新JSON记录
        new_item = item.copy()
        new_item["img_path"] = new_img_rel_path
        new_item["mask_path"] = new_mask_rel_path  # 写入新mask的路径
        new_data.append(new_item)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=4)

    print(f"\n处理完成！新的JSON文件已保存至: {output_json_path}")


# ==============================================================================
# 启动入口
# ==============================================================================
if __name__ == '__main__':

    # ==========================================================================
    # --- 1. 在这里修改您的配置 ---
    # ==========================================================================

    # 您项目的数据根目录
    # 脚本会把这个路径与JSON中的相对路径拼接起来
    DATA_ROOT = "dataset"

    # 您想要处理的一个或多个JSON文件列表
    # 注意：这里的路径是相对于您运行脚本的位置的路径
    INPUT_JSON_FILES = [
        "dataset/processed_train.json",
        "dataset/processed_test.json"
        # "another_dataset.json",
    ]

    # ==========================================================================
    # --- 配置结束，下方代码通常无需修改 ---
    # ==========================================================================

    print("正在加载人脸检测模型...")
    prototxt_path = "model_parameter/deploy.prototxt.txt"
    model_path = "model_parameter/res10_300x300_ssd_iter_140000.caffemodel"
    if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
        print(f"错误：找不到人脸检测模型文件。脚本将退出。")
        exit()
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    print("模型加载成功。")

    for input_json_file in INPUT_JSON_FILES:
        if not os.path.exists(input_json_file):
            print(f"错误：配置的JSON文件 '{input_json_file}' 不存在，已跳过。")
            continue

        base, ext = os.path.splitext(input_json_file)
        output_json_file = f"{base}_cut{ext}"

        print(f"\n" + "=" * 50)
        print(f"开始处理: {input_json_file}")
        print(f"数据根目录: {DATA_ROOT}")
        print(f"输出JSON将保存至: {output_json_file}")
        print(f"=" * 50)

        process_json_dataset(
            input_json_path=input_json_file,
            output_json_path=output_json_file,
            data_root=DATA_ROOT,
            net=net
        )