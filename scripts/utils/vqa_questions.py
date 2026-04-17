from lavis.models import load_model_and_preprocess
import torch
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)
# ask a random question.

raw_image = Image.open("/home/share/qrn8y2ug/home/ysuanAJ27/ljc/LAVIS-main/dataset/dataset/celeb512/mat_gan/samples_celeb/20936.png")

# question = "This face is a collage made from different other faces. Please list all specific parts of the face that are collaged, such as eyes, nose, mouth, ears, skin texture, hair, and clothing. Be as detailed as possible."
question = "This face is synthesized by AI. Please list all parts of this face that are synthesized by AI, such as the beard, clothes, etc."
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
question = txt_processors["eval"](question)
answers = model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate", max_length=60)
# answers = model.predict_answers(
#     samples={"image": image, "text_input": question},
#     inference_method="generate",
#     num_beams=1  # 禁用 beam-search
# )

# 降级transformers到4.26

print(answers)