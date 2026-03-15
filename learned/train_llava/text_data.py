import json
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import LlavaProcessor,LlavaForConditionalGeneration


import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import os
#from evaluate_with_attn_vis import evaluate_model_with_attn_vis

import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import AutoProcessor

from peft import PeftModel

import torch

class LlavaTestDataset(Dataset):
    def __init__(self, dataset_dir: str,device : torch.device) -> None:
        super().__init__()
        self.device = device
        self.chat_data, self.image_dir = self.build_dataset(dataset_dir)

    def build_dataset(self, data_dir: str) -> Tuple[List[Dict], Path]:
        data_dir = Path(data_dir)
        chat_file = data_dir / "juice_bottle/strcutural.json" # 测试数据 JSON 文件
        image_dir = data_dir / "juice_bottle/structural_anomalies"  # 图像文件夹
        #加载josn数据为字典列表
        chat_data = pd.read_json(chat_file).to_dict(orient="records")

        return chat_data, image_dir

    def __len__(self):
        return len(self.chat_data)

    def __getitem__(self, index) -> Tuple[str, Path, str]:
        cur_data = self.chat_data[index]
        conversations = cur_data.get("conversations")

        # 提取人类问题和图像路径
        q_text = conversations[0].get("value")
        image_path = self.image_dir / cur_data.get("image")
        true_anwser = conversations[1].get("value")
        anomaly_status = cur_data.get("anomaly_info").get("is_anomalous")

        # 提取真实标签
        return q_text, image_path, anomaly_status,true_anwser
    

def build_test_input(processor: AutoProcessor, q_text: str, image_path: Path):
    """
    构建测试阶段的输入，仅包括 input_ids 和 pixel_values。
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": q_text},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    raw_image = Image.open(image_path)

    inputs = processor(prompt, raw_image, return_tensors="pt",padding = True,truncation = True,max_length = 512)
    return inputs["input_ids"], inputs["pixel_values"]


def evaluate_model(processor: AutoProcessor, lora_model, test_dataset: LlavaTestDataset,output_file: str,
                   device: torch.device
                   ):
    results = []
    correct_predictions = 0 
 
    for i in range(len(test_dataset)):
        q_text, image_path, anomaly_status, true_anwser = test_dataset[i]
    # print(test_dataset[2])
    # 构建测试输入
        input_ids, pixel_values = build_test_input(processor, q_text, image_path)
        input_ids, pixel_values = input_ids.to(device), pixel_values.to(device)
    # 模型预测
        outputs = lora_model.generate(
            input_ids=input_ids, 
            pixel_values=pixel_values,
            attention_mask = torch.ones_like(input_ids),
            pad_token_id= processor.tokenizer.pad_token_id,
            max_new_tokens=100,
            use_cache = True,
            )

        predicted_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

        #print(f"ID: {i}, Predicted Text: {predicted_text}, Predicted Status: {predicted_status}, True Status: {anomaly_status}, Correct: {is_correct}")
        #predicted_text = "".join(predicted_text.split())
        #print("预测的文本:",predicted_text)
        results.append({
            "id": i,
            "question": q_text,
            "predicted_answer": predicted_text,
            "anomaly_status": anomaly_status,  # 真实标签
            "image": str(image_path),
            "true_anwser": true_anwser,
            "correct_predictions": correct_predictions
        
        })

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    return results



if __name__ == "__main__":

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # 加载基础模型
    base_model = LlavaForConditionalGeneration.from_pretrained(
        "/home/yuxuan/yuanxuan-new/show_model/model001"
    ).to(device)
    print("加载基础模型成功")
    # 加载 LoRA 权重
    lora_model = PeftModel.from_pretrained(
        base_model,
        "/home/yuxuan/yuanxuan-new/result/juice-bottle/train2new-output/output_model_user_lora_0514/checkpoint-240"
    ).to(device)
    print("加载lora模型成功")
    # 加载 AutoProcessor
    processor = LlavaProcessor.from_pretrained("/home/yuxuan/yuanxuan-new/show_model/model001")
    print(
        "加载process成功！"
    )
    # 测试数据集路径

    test_data_dir = "/home/yuxuan/yuanxuan-new/text_data"
    test_dataset = LlavaTestDataset(test_data_dir,device)
   
    # 结果输出
    output_file = "/home/yuxuan/yuanxuan-new/result/juice-bottle/train2new-output/result_strcutural.json"

    results = evaluate_model(processor, lora_model, test_dataset,output_file,device)

    #打印第一个结果
    if results:
        res = results[0]
        print(f"ID: {res['id']}, Predicted: {res['predicted_answer']}, True: {res['anomaly_status']}")
    else:
        print("没有生成任何结果。")

