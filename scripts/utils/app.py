import gradio as gr
import torch
import numpy as np
from PIL import Image
import time

# LAVIS 相关依赖
from omegaconf import OmegaConf
from lavis.common.registry import registry
from lavis.models import load_preprocess

# ==========================================
# ⚠️ 请在这里替换为你真实的权重路径
# ==========================================
TEACHER_CKPT = "/scratch-shared/ljc-1/dff/model_parameter/2111_checkpoint_best.pth" 
STUDENT_CKPT = "/scratch-shared/ljc-1/dff/model_parameter/checkpoint_best.pth" # 你刚才那个 0.482 的权重路径
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher_model = None
student_model = None
vis_processors = None

def build_offline_model(model_name, model_type, ckpt_path):
    """
    专门针对集群计算节点的离线模型加载器，绕过外网下载。
    """
    # 1. 获取模型类和默认配置路径
    model_cls = registry.get_model_class(model_name)
    cfg_path = model_cls.default_config_path(model_type)
    
    # 2. 加载配置并强行关闭预训练权重下载
    cfg = OmegaConf.load(cfg_path)
    cfg.model.pretrained = ""          
    cfg.model.finetuned = ""           
    cfg.model.load_pretrained = False  
    cfg.model.load_finetuned = False   
    
    # 3. 初始化纯净的模型骨架
    model = model_cls.from_config(cfg.model)
    
    # 4. 加载你自己的本地权重
    print(f"   -> 正在从本地注入权重: {ckpt_path}")
    model.load_checkpoint(ckpt_path)
    model = model.eval().to(device)
    
    # 5. 加载图像处理器
    v_processors, _ = load_preprocess(cfg.preprocess)
    return model, v_processors

def init_models():
    global teacher_model, student_model, vis_processors
    print(f"🚀 [Init] 正在 H100 上执行离线安全加载模式，请稍候...")
    
    # 1. 加载 Teacher 模型 (XL)
    print("📥 正在加载重型教师模型 (FLAN-T5-XL)...")
    teacher_model, vis_processors = build_offline_model("blip2_t5_instruct", "flant5xl", TEACHER_CKPT)
    
    # 2. 加载 Student 模型 (Base)
    print("📥 正在加载轻量化学生模型 (FLAN-T5-Base)...")
    student_model, _ = build_offline_model("blip2_t5_instruct", "flant5base", STUDENT_CKPT)
    
    print("✅ 双模型本地加载完毕！UI 准备就绪。")

def overlay_mask_on_image(img_pil, mask_tensor, threshold=0.5, alpha=0.5, color=(255, 0, 0)):
    """
    将模型输出的掩码张量，以半透明颜色叠加到原始 PIL 图像上
    """
    mask_np = mask_tensor.squeeze().cpu().float().numpy() 
    mask_binary = (mask_np > threshold).astype(np.uint8)
    
    mask_img = Image.fromarray(mask_binary * 255, mode='L')
    mask_img = mask_img.resize(img_pil.size, resample=Image.NEAREST)
    mask_resized_np = np.array(mask_img) / 255.0
    
    img_np = np.array(img_pil).astype(np.float32)
    colored_layer = np.zeros_like(img_np)
    colored_layer[:] = color
    
    blended = img_np.copy()
    mask_indices = mask_resized_np == 1
    blended[mask_indices] = img_np[mask_indices] * (1 - alpha) + colored_layer[mask_indices] * alpha
    
    return Image.fromarray(blended.astype(np.uint8))

@torch.no_grad()
def run_inference(input_img, run_teacher, run_student):
    if input_img is None:
        return None, "请先上传图像！", None, "请先上传图像！"
    
    image_tensor = vis_processors["eval"](input_img).unsqueeze(0).to(device)
    
    t_text, t_img = "未选择执行", None
    s_text, s_img = "未选择执行", None
    
    if run_teacher:
        start_time = time.time()
        # 注意：这里调用你修改后的 generate 方法，返回 (texts, masks)
        t_texts, t_masks = teacher_model.generate({"image": image_tensor}, max_length=100)
        t_cost = time.time() - start_time
        t_text = f"⏱️ 推理耗时: {t_cost:.2f} 秒\n\n📝 归因报告:\n{t_texts[0]}"
        t_img = overlay_mask_on_image(input_img, t_masks[0], color=(255, 0, 0)) # Teacher 红色掩码
        
    if run_student:
        start_time = time.time()
        s_texts, s_masks = student_model.generate({"image": image_tensor}, max_length=100)
        s_cost = time.time() - start_time
        s_text = f"⏱️ 推理耗时: {s_cost:.2f} 秒\n\n📝 归因报告:\n{s_texts[0]}"
        s_img = overlay_mask_on_image(input_img, s_masks[0], color=(0, 255, 0)) # Student 绿色掩码

    return t_img, t_text, s_img, s_text

# ==========================================
# 启动机制
# ==========================================
if __name__ == "__main__":
    init_models()
    
    with gr.Blocks(theme=gr.themes.Soft(), title="ForgeryTalker 预答辩演示") as demo:
        gr.Markdown(
            """
            # 🛡️ ForgeryTalker: 生成式面部伪造多模态主动归因系统
            **模型规模对比测试：** 重型教师模型 (5.0B) VS 轻量化蒸馏学生模型 (2.3B)
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="1. 上传待检测图像")
                run_teacher_cb = gr.Checkbox(label="启用 Teacher 模型", value=True)
                run_student_cb = gr.Checkbox(label="启用 Student 模型", value=True)
                submit_btn = gr.Button("🚀 执行端到端推断", variant="primary")
                
            with gr.Column(scale=2):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🎓 重型教师模型 (FLAN-T5-XL)")
                        teacher_mask_out = gr.Image(label="物理空间伪影锚定")
                        teacher_text_out = gr.Textbox(label="语义归因逻辑溯源", lines=5)
                        
                    with gr.Column():
                        gr.Markdown("### 🎒 轻量化学生模型 (FLAN-T5-Base)")
                        student_mask_out = gr.Image(label="物理空间伪影锚定")
                        student_text_out = gr.Textbox(label="语义归因逻辑溯源", lines=5)

        submit_btn.click(
            fn=run_inference,
            inputs=[input_image, run_teacher_cb, run_student_cb],
            outputs=[teacher_mask_out, teacher_text_out, student_mask_out, student_text_out]
        )

    # 启动 Gradio
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)