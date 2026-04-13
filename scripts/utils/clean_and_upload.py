import torch
import os
from huggingface_hub import HfApi

# 1. 路径设置
ckpt_path = "/gpfs/scratch1/shared/ljc-1/dff/lavis/output/instructBlip_student_lr1e4_stage2_finetune/20260313090/checkpoint_best.pth"
save_path = "./pytorch_model.bin" # 洗澡+抽水后的纯净权重，HF标准命名

print("⏳ 正在加载原始 Checkpoint...")
ckpt = torch.load(ckpt_path, map_location="cpu")

# 2. 剥离 optimizer, scaler, epoch 等杂质
state_dict = ckpt['model'] if 'model' in ckpt else ckpt

# 3. 核心操作：FP32 -> FP16 (体积减半绝招！)
print("🗜️ 正在将模型精度从 FP32 转换为 FP16 (瘦身 50%)...")
fp16_state_dict = {}
for k, v in state_dict.items():
    fp16_state_dict[k] = v.half()

# 4. 保存本地
print(f"💾 正在保存 FP16 纯净权重到 {save_path}...")
torch.save(fp16_state_dict, save_path)
size_mb = os.path.getsize(save_path) / (1024 * 1024)
print(f"✅ 瘦身成功！新文件大小约为: {size_mb:.2f} MB")

# ==========================================
# 5. 上传到 Hugging Face (确保你已经用 huggingface-cli login 登录过)
# ==========================================
api = HfApi()

# 【请修改】：换成你的 HF 用户名和想要的仓库名
repo_id = "LianJC/DFF-T5Base-Forgery-Student" 

print(f"\n📦 正在 Hugging Face 上创建/定位仓库: {repo_id} ...")
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

print("🚀 正在上传模型权重，请耐心等待...")
api.upload_file(
    path_or_fileobj=save_path,
    path_in_repo="pytorch_model.bin",
    repo_id=repo_id,
    repo_type="model"
)

print(f"🎉 恭喜！上传完成！你的轻量级蒸馏模型现在已经挂在：https://huggingface.co/{repo_id}")
