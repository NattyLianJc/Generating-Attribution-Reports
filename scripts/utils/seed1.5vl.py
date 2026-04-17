import os
import json
import base64
from openai import OpenAI # 或者您使用的火山引擎 Ark 客户端

# --- 配置部分 ---
# 0. 是否只测试单条记录
TEST_SINGLE_RECORD = False  # <<<< 设置为 True 则只处理第一条记录，False 处理所有记录

# 1. 您的 JSON 文件路径
JSON_FILE_PATH = 'dataset/eval-mutilabel.json'

# 2. 图片文件所在的顶层目录
TOP_LEVEL_IMAGE_DIR = 'dataset' # 例如 'dataset' 或 '/path/to/your/image_root_folder'

# 3. 输出文件名
OUTPUT_JSON_FILENAME_BASE = 'api_analysis_results'

# 4. 从环境变量中获取 API Key
ARK_API_KEY = os.environ.get("ARK_API_KEY")
if not ARK_API_KEY:
    print("错误：请设置环境变量 ARK_API_KEY。")
    exit()

# 5. 初始化 Ark 客户端
client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=ARK_API_KEY,
)

# 6. 精简后的英文提示词
NEW_ENGLISH_PROMPT = "Analyze the AI-altered facial image. Provide a concise, single-paragraph description of the observed modifications. For instance, an output could be: \"The subject's eyes exhibit diverse shapes and sizes, with the right one appearing smaller, and variations in sclera and pupil colors. The face shows asymmetry, particularly the nose, which looks unrealistic.\" Your description should focus on observable changes to skin texture, facial symmetry, eye details, nasal structure, lip morphology, and any unnatural artifacts or blending."

# 7. 您指定的模型 ID
MODEL_ID = "doubao-1-5-thinking-vision-pro-250428"

# --- 辅助函数 (保持不变) ---
def get_image_mime_type(file_path):
    """根据文件扩展名推断 MIME 类型 (中文注释)"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".png": return "image/png"
    elif ext in [".jpg", ".jpeg"]: return "image/jpeg"
    elif ext == ".gif": return "image/gif"
    elif ext == ".webp": return "image/webp"
    else:
        print(f"警告: 未知图片扩展名 {ext} (文件: {file_path})，将尝试使用 image/jpeg。")
        return "image/jpeg"

def image_to_base64_data_url(file_path):
    """将本地图片文件转换为 Base64 Data URL (中文注释)"""
    try:
        mime_type = get_image_mime_type(file_path)
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    except FileNotFoundError:
        print(f"错误：图片文件未找到 {file_path}")
        return None
    except Exception as e:
        print(f"错误：转换图片 {file_path} 到 Base64 失败: {e}")
        return None

# --- 函数：加载和处理数据 ---
def load_and_prepare_messages(json_file_path, validated_top_level_img_dir, text_prompt, process_limit=None):
    """
    从 JSON 文件加载数据，根据顶层目录和JSON中的相对路径定位图片，
    将其转换为 Base64 Data URL，并准备 API 调用所需的 messages 结构。(中文注释)
    process_limit: 可选参数，限制处理的记录数量。
    """
    all_prepared_items = []
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            records_from_json = json.load(f)
    except FileNotFoundError:
        print(f"错误：JSON 文件 {json_file_path} 未找到。")
        return []
    except json.JSONDecodeError:
        print(f"错误：JSON 文件 {json_file_path} 不是有效的 JSON 格式。")
        return []
    except Exception as e:
        print(f"错误：读取JSON文件 {json_file_path} 时发生意外错误: {e}")
        return []

    if not isinstance(records_from_json, list):
        print(f"错误：JSON 文件的根元素应该是一个列表。当前在 {json_file_path} 中找到的类型是: {type(records_from_json)}")
        return []

    records_to_process = records_from_json
    if process_limit is not None and isinstance(process_limit, int) and process_limit > 0:
        if process_limit < len(records_from_json):
            print(f"信息：将处理记录数量限制为前 {process_limit} 条。")
        records_to_process = records_from_json[:process_limit]
    elif not records_from_json:
        print("信息：JSON 文件中没有记录可处理。")
        return []


    for record_index, original_record in enumerate(records_to_process):
        if not isinstance(original_record, dict):
            print(f"警告：JSON记录索引 {record_index} 不是字典格式，已跳过: {original_record}")
            continue

        relative_img_path_from_json = original_record.get("img_path")

        if not relative_img_path_from_json:
            print(f"警告：记录索引 {record_index} (ID: {original_record.get('image_id', '未知ID')}) 缺少 'img_path' 字段，已跳过。")
            continue

        absolute_img_path = os.path.normpath(os.path.join(validated_top_level_img_dir, relative_img_path_from_json))
        image_data_url = image_to_base64_data_url(absolute_img_path)

        if not image_data_url:
            print(f"警告：无法处理图片 {absolute_img_path} (来自记录索引 {record_index}, ID: {original_record.get('image_id', '未知ID')})，已跳过。")
            continue

        message_content = [
            {"type": "image_url", "image_url": {"url": image_data_url}},
            {"type": "text", "text": text_prompt},
        ]
        api_message_format = [{"role": "user", "content": message_content}]

        all_prepared_items.append({
            "messages": api_message_format,
            "original_path": absolute_img_path,
            "original_record_data": original_record # 保存原始记录数据
        })

    return all_prepared_items

# --- 主逻辑 ---
if __name__ == "__main__":
    # 1. 校验和规范化 TOP_LEVEL_IMAGE_DIR
    if not TOP_LEVEL_IMAGE_DIR:
        print(f"错误：'TOP_LEVEL_IMAGE_DIR' 未配置。请设置图片所在的顶层目录。")
        exit()
    # 如果用户忘记修改占位符，这里给出提示
    if TOP_LEVEL_IMAGE_DIR == '/path/to/your/top_level_image_directory/':
         print(f"错误：请在脚本中更改占位符 'TOP_LEVEL_IMAGE_DIR' ('{TOP_LEVEL_IMAGE_DIR}') 为您实际的顶层图片目录。")
         exit()

    abs_top_level_img_dir = os.path.abspath(TOP_LEVEL_IMAGE_DIR)
    if not os.path.isdir(abs_top_level_img_dir):
        print(f"错误：解析后的顶层图片目录 '{abs_top_level_img_dir}' (来自配置 '{TOP_LEVEL_IMAGE_DIR}') 不是一个有效的目录或不存在。")
        exit()
    print(f"使用的顶层图片目录 (绝对路径): {abs_top_level_img_dir}")

    # 2. 根据 TEST_SINGLE_RECORD 设置处理限制和输出文件名
    record_limit = 1 if TEST_SINGLE_RECORD else None
    output_filename_suffix = "_TEST" if TEST_SINGLE_RECORD else ""
    final_output_json_file = f"{OUTPUT_JSON_FILENAME_BASE}{output_filename_suffix}.json"

    if TEST_SINGLE_RECORD:
        print("--- 模式：单条记录测试 ---")

    # 3. 加载并准备数据
    all_prepared_data_items = load_and_prepare_messages(
        JSON_FILE_PATH,
        abs_top_level_img_dir,
        NEW_ENGLISH_PROMPT,
        process_limit=record_limit
    )

    if not all_prepared_data_items:
        print("没有可供处理的数据（可能JSON文件为空或记录无法处理），程序退出。")
        exit()

    print(f"成功为 {len(all_prepared_data_items)} 条记录准备了 API 调用数据。")

    # 4. 遍历准备好的数据，并为每个图片调用 API
    all_api_responses_to_save = []
    for i, data_item in enumerate(all_prepared_data_items):
        messages_for_one_image = data_item["messages"]
        original_image_path = data_item["original_path"]
        original_record = data_item["original_record_data"]

        print(f"\n正在处理第 {i+1}/{len(all_prepared_data_items)} 张图片: {original_image_path}")
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages_for_one_image,
                temperature=0.5,
                max_tokens=1024
            )
            print(f"  API 调用成功。")
            response_content = ""
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                 response_content = response.choices[0].message.content
                 print(f"  响应预览: {response_content[:100]}...")
            else:
                print(f"  响应中未找到预期的 choices 或 message.content 结构。")

            all_api_responses_to_save.append({
                "status": "success",
                "original_data": original_record,
                "image_path_processed": original_image_path,
                "api_response_text": response_content,
                "full_api_response": response.to_dict() if hasattr(response, "to_dict") else str(response)
            })
        except Exception as e:
            print(f"  API 调用失败: {e}")
            all_api_responses_to_save.append({
                "status": "error",
                "original_data": original_record,
                "image_path_processed": original_image_path,
                "error_message": str(e)
            })

    # 5. API 响应总结并保存到文件
    print("\n--- API 调用总结 ---")
    success_count = sum(1 for r in all_api_responses_to_save if r["status"] == "success")
    error_count = len(all_api_responses_to_save) - success_count
    print(f"总计: {success_count} 次成功, {error_count} 次失败。")

    with open(final_output_json_file, 'w', encoding='utf-8') as f_out:
        json.dump(all_api_responses_to_save, f_out, ensure_ascii=False, indent=4)
    print(f"完整的 API 响应和分析结果已保存到 {final_output_json_file}")