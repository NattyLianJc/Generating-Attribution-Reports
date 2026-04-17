import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="/home/share/qrn8y2ug/home/ysuanAJ27/ljc/LAVIS-main/model-parameter/TowerInstruct-7B-v0.2", torch_dtype=torch.float32, device_map="auto")
# We use the tokenizer’s chat template to format each message - see https://hf-mirror.com/docs/transformers/main/en/chat_templating
messages = [
    {"role": "user", "content": "精简这个句子，还是保持原语言英文：The woman's mouth was deformed, and her teeth were unclear, and her eyelashes were unclear. Her mouth was deformed, and her upper lip had no texture, and her neck skin and face skin were inconsistent, and the contour of her hairline was unclear. Part of her right eyebrow was missing, and her eyes were deformed."},
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=80, do_sample=False)
print(outputs[0]["generated_text"])
# <|im_start|>user
# Translate the following text from Portuguese into English.
# Portuguese: Um grupo de investigadores lançou um novo modelo para tarefas relacionadas com tradução.
# English:<|im_end|>
# <|im_start|>assistant
# A group of researchers has launched a new model for translation-related tasks.