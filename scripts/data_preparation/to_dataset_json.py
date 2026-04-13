import json
import os

# 文件路径配置（根据实际情况修改）
eval_path = "dataset/eval-mutilabel.json"
train_path = "dataset/train-mutilabel.json"
test_path = "dataset/test-mutilabel.json"
new_file_path = "dataset/face_attribute_dataset_mutilabel.json"  # 新 json 文件路径

# 读取原始 JSON 文件
with open(eval_path, 'r', encoding='utf-8') as f:
    eval_data = json.load(f)

with open(train_path, 'r', encoding='utf-8') as f:
    train_data = json.load(f)

with open(test_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# 统计当前三个文件中的条目数
eval_count = len(eval_data)
train_count = len(train_data)
test_count = len(test_data)
total_existing = eval_count + train_count + test_count

print(f"当前文件条目数：eval={eval_count}, train={train_count}, test={test_count}")

# 计算每个文件的比例
prop_eval = eval_count / total_existing
prop_train = train_count / total_existing
prop_test = test_count / total_existing

# 读取新 JSON 文件中的所有条目
with open(new_file_path, 'r', encoding='utf-8') as f:
    new_data = json.load(f)

new_total = len(new_data)
print(f"新数据总条目数：{new_total}")

# 按照比例计算新数据分配的数量（初始采用整数部分）
num_eval = int(new_total * prop_eval)
num_train = int(new_total * prop_train)
num_test = int(new_total * prop_test)

# 计算分配后的总数，若不足则根据各自的小数余数补充（防止舍入损失）
assigned = num_eval + num_train + num_test
remain = new_total - assigned

# 计算各部分的小数余数
frac_eval = (new_total * prop_eval) - num_eval
frac_train = (new_total * prop_train) - num_train
frac_test = (new_total * prop_test) - num_test

# 构建列表，并按照余数由大到小排序
fractions = [("eval", frac_eval), ("train", frac_train), ("test", frac_test)]
fractions.sort(key=lambda x: x[1], reverse=True)

i = 0
while remain > 0:
    key, _ = fractions[i]
    if key == "eval":
        num_eval += 1
    elif key == "train":
        num_train += 1
    elif key == "test":
        num_test += 1
    remain -= 1
    i = (i + 1) % len(fractions)

print(f"新数据分配数量：eval={num_eval}, train={num_train}, test={num_test}")

# 检查分配数之和是否等于总数
assert num_eval + num_train + num_test == new_total, "分配数量不等于新数据总数！"

# 按顺序将新数据中的条目切分成三部分
eval_new_items = new_data[:num_eval]
train_new_items = new_data[num_eval: num_eval + num_train]
test_new_items = new_data[num_eval + num_train: num_eval + num_train + num_test]

# 将新条目追加到原始数据中
eval_data.extend(eval_new_items)
train_data.extend(train_new_items)
test_data.extend(test_new_items)

# 写回更新后的 JSON 文件
with open(eval_path, 'w', encoding='utf-8') as f:
    json.dump(eval_data, f, indent=4, ensure_ascii=False)

with open(train_path, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)

with open(test_path, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=4, ensure_ascii=False)

print("新数据已按比例追加到三个 JSON 文件中。")
