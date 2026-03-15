import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from transformers import LlavaProcessor,LlavaForConditionalGeneration
from peft import PeftModel
import torch
from PIL import Image
import os

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def generate_and_trace_all_tokens(
    model, processor, q_text, image_path, device,
    max_new_tokens=50, image_token_count=576, save_dir="/home/yuxuan/yuanxuan-new/text_data/breakfast-box/attentionmap"
):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    model.config.output_attentions = True

    # 构建 prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": q_text}
    ]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(prompt, image, return_tensors="pt").to(device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]

 # 打印调试信息，确认图像 token 插入成功
    image_token_id = model.config.image_token_index
    image_token_count_in_input = (input_ids == image_token_id).sum().item()
    print(f"🧪 图像 token ID ({image_token_id}) 出现次数: {image_token_count_in_input}（期望为576）")

    special_tokens = processor.tokenizer.special_tokens_map
    print("🚧 special_tokens_map:", special_tokens)
    print("🧪 是否定义了图像 patch token：", "<image_patch_token>" in processor.tokenizer.get_vocab())


    generated = input_ids
    all_generated_tokens = []

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=generated,
                attention_mask=torch.ones_like(generated),
                pixel_values=pixel_values,
                use_cache=False,
                output_attentions=True,
                return_dict=True
            )

        # 获取 logits & 生成下一个 token
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        generated = torch.cat([generated, next_token], dim=-1)

        # 记录生成 token
        generated_token = processor.tokenizer.convert_ids_to_tokens(next_token[0])[0]
        all_generated_tokens.append(generated_token)

        # 提取 attention（最后一层 self-attention）
        last_attn = outputs.attentions[-1][0]  # [num_heads, seq_len, seq_len]
        image_attn = last_attn[:, -1, :image_token_count]  # [num_heads, image_tokens]
        mean_attn = image_attn.mean(dim=0).cpu().numpy().reshape(24, 24)

        # 可视化并保存
        plt.figure(figsize=(4, 4))
        plt.imshow(mean_attn, cmap="hot")
        plt.colorbar()
        plt.title(f"Token: {generated_token}")
        save_path = os.path.join(save_dir, f"step_{step+1:02d}_{generated_token.replace('/', '_')}.png")
        print(f"🔍 正在保存 attention 图像: {save_path}")
        plt.savefig(save_path)
        plt.close()

    # 输出完整生成文本
    decoded = processor.tokenizer.decode(generated[0], skip_special_tokens=True)
    print("Generated Text:", decoded)
    print("Generated Tokens:", all_generated_tokens)
    print(f"Saved attention visualizations to: {save_dir}")

if __name__ == "__main__":

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # 加载基础模型
    base_model = LlavaForConditionalGeneration.from_pretrained(
        "/home/yuxuan/yuanxuan-new/show_model/model001"
    ).to(device)
    print("加载基础模型成功")


    # 加载 LoRA 权重
    model = lora_model = PeftModel.from_pretrained(
        base_model,
        "/home/yuxuan/yuanxuan-new/result/juice-bottle/train2-output/output_model_user_lora_0115/checkpoint-900"
    ).to(device)
    print("加载lora模型成功")
    print("当前模型类型:", type(lora_model))
    print("底层模型类型:", type(lora_model.base_model.model))


    # 加载 AutoProcessor
    processor = LlavaProcessor.from_pretrained("/home/yuxuan/yuanxuan-new/show_model/model001")
    print(
        "加载process成功！"
    )

    # 设置输入
    prompt = "Is there any anomaly in this image? <image>"
    image_path = "/home/yuxuan/yuanxuan-new/text_data/juice_bottle/logical_anomalies/001_logic.png"
    #device = torch.device("cuda:2")
    #save_dir = "/home/yuxuan/yuanxuan-new/text_data/breakfast-box/attentionmap"
    # 可视化 attention
    image = Image.open(fp=image_path)
    inputs = processor(text=prompt, images = image, return_tensors="pt")
    for temp_key in inputs.keys():
        inputs[temp_key] = inputs[temp_key].to(device)
    generate_ids = model.generate(**inputs, max_new_tokens = 15)
    respones = processor.batch_decode(generate_ids, skip_special_tokens = True,clean_up_tokenization_spaces = False)[0]
    print(respones)
    #generate_and_trace_all_tokens(
    #    model=lora_model,
    #    processor=processor,
    #   q_text=q_text,
    #   image_path=image_path,
    #    device=torch.device("cuda:2"),
    #    save_dir=save_dir # 可改为你要分析的 token
    #)
