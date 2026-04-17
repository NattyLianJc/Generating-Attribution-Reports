import json
import pandas as pd
import numpy as np
import re
from collections import Counter


def analyze_facial_data(file_paths):
    """
    加载、分类并分析面部图像描述数据（增加总词数和平均词数统计）。

    Args:
        file_paths (list): 包含JSON文件路径的列表。
    """
    # 1. 加载并合并数据
    all_data = []
    print("正在加载文件...")
    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                all_data.extend(json.load(f))
            print(f"  - 成功加载: {path}")
        except FileNotFoundError:
            print(f"错误：文件未找到 '{path}'。请确保脚本相对于'dataset'文件夹的路径正确。")
            return
        except json.JSONDecodeError:
            print(f"错误：无法解析JSON文件 '{path}'。请检查文件格式。")
            return

    if not all_data:
        print("错误：未加载任何数据。")
        return

    df = pd.DataFrame(all_data)

    # 2. 定义分类函数并应用
    def categorize_image_id(image_id):
        if image_id.startswith('sd'):
            return 'Diff. Inp.'
        elif image_id.startswith('mat'):
            return 'Trans. Inp.'
        elif image_id.startswith('e4s'):
            return 'GAN-FS'
        else:
            return 'FAE'

    df['category'] = df['image_id'].apply(categorize_image_id)

    # 统计单词数量
    df['caption_length'] = df['caption'].str.split().str.len()

    print("\n数据分类完成。各类别样本数：")
    print(df['category'].value_counts())

    # ==========================
    # ===   【核心修改处】   ===
    # ==========================
    # 计算并打印总词数和平均词数
    total_word_count = df['caption_length'].sum()
    average_word_count = df['caption_length'].mean()

    print("\n" + "=" * 50)
    print("全局文本统计:")
    # 使用 f-string 的格式化功能，为总数加上千位分隔符，平均数保留两位小数
    print(f"  所有 Caption 的总词数: {total_word_count:,}")
    print(f"  所有 Caption 的平均词数: {average_word_count:.2f}")
    print("=" * 50 + "\n")

    # 3. Caption 长度分段统计 (保持不变)
    print("Caption 长度统计 (按单词数):")
    categories = ['Diff. Inp.', 'Trans. Inp.', 'GAN-FS', 'FAE']
    for category in categories:
        print(f"\n--- 类别: {category} ---")
        category_df = df[df['category'] == category]
        if not category_df.empty:
            min_len = category_df['caption_length'].min()
            max_len = category_df['caption_length'].max()
            print(f"最少单词数: {min_len}")
            print(f"最多单词数: {max_len}")

            bin_width = 5
            upper_bound = (max_len // bin_width + 2) * bin_width
            bins = np.arange(0, upper_bound, bin_width)

            hist, edges = np.histogram(category_df['caption_length'], bins=bins)
            print(f"长度频率分布 (区间长度为{bin_width}个单词):")
            for i in range(len(hist)):
                print(f"  单词数 {edges[i]}-{edges[i + 1] - 1}: {hist[i]}")
        else:
            print("该类别下没有数据。")
    print("\n" + "=" * 50 + "\n")

    # 4. 修改区域统计 (保持不变)
    # ... (此处省略与上一版完全相同的修改区域统计代码，以保持简洁) ...
    # 为了简洁，这里不再重复粘贴，实际运行时请使用包含这部分完整逻辑的代码
    facial_features = {
        "eye": ["eye", "eyes", "ocular", "optic", "vision"], "pupil": ["pupil", "pupils"],
        "lip": ["lip", "lips", "labium", "labia"], "mouth": ["mouth", "mouths", "oral cavity", "buccal"],
        "nose": ["nose", "noses", "nasal", "nostril", "nostrils"], "ear": ["ear", "ears", "auricle", "pinna", "pinnae"],
        "cheek": ["cheek", "cheeks", "buccal", "malar"], "forehead": ["forehead", "frontal", "brow", "brows"],
        "chin": ["chin", "chins", "mentum"], "jaw": ["jaw", "jaws", "mandible", "maxilla"],
        "eyebrow": ["eyebrow", "eyebrows"], "tooth": ["tooth", "teeth", "dentition"],
        "tongue": ["tongue", "tongues", "glossa", "lingua"], "beard": ["beard", "beards", "facial hair"],
        "mustache": ["mustache", "mustaches", "moustache", "moustaches"],
        "skin": ["skin", "derma", "epidermis", "cutis"],
        "hair": ["hair", "hairs", "locks", "tresses", "strands"],
        "wrinkle": ["wrinkle", "wrinkles", "line", "lines", "crease", "creases"],
        "freckle": ["freckle", "freckles", "lentigo", "lentigines"],
        "scar": ["scar", "scars", "cicatrix", "cicatrices"],
        "dimple": ["dimple", "dimples", "fossa", "fossae"]
    }
    synonym_to_feature = {synonym.lower(): feature for feature, synonyms in facial_features.items() for synonym in
                          synonyms}

    def count_modified_features_in_caption(caption, synonym_map):
        found_features = set()
        all_synonyms_pattern = r'\b(' + '|'.join(re.escape(s) for s in synonym_map.keys()) + r')\b'
        matches = re.findall(all_synonyms_pattern, caption.lower())
        for match in matches:
            feature = synonym_map[match]
            found_features.add(feature)
        return len(found_features)

    print("修改区域统计:")
    analysis_categories = ['Diff. Inp.', 'Trans. Inp.', 'FAE']
    for category in analysis_categories:
        print(f"\n--- 类别: {category} ---")
        category_df = df[df['category'] == category].copy()
        if not category_df.empty:
            per_sample_counts = category_df['caption'].apply(
                lambda x: count_modified_features_in_caption(x, synonym_to_feature))
            print("每个样本的修改区域个数统计:")
            counts_distribution = per_sample_counts.value_counts().sort_index()
            if counts_distribution.empty or counts_distribution.sum() == 0:
                print("  未在该类别中发现任何指定的面部特征。")
            else:
                for num_areas, num_samples in counts_distribution.items():
                    if num_areas > 0: print(f"  修改了 {num_areas} 个区域的样本数: {num_samples}")
            total_category_mentions = Counter()
            all_synonyms_pattern = r'\b(' + '|'.join(re.escape(s) for s in synonym_to_feature.keys()) + r')\b'
            for caption in category_df['caption']:
                matches = re.findall(all_synonyms_pattern, caption.lower())
                for match in matches:
                    feature = synonym_to_feature[match]
                    total_category_mentions[feature] += 1
            print("\n该类别下各特征被修改的总次数 (降序):")
            if not total_category_mentions:
                print("  未在该类别中发现任何指定的面部特征。")
            else:
                for feature, count in total_category_mentions.most_common(): print(f"  {feature}: {count}")
        else:
            print("该类别下没有数据。")


# --- 使用说明 ---
json_files = ["dataset/train-mutilabel.json", "dataset/eval-mutilabel.json", "dataset/test-mutilabel.json"]

analyze_facial_data(json_files)