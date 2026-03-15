import argparse
import base64
import json

import requests
import cv2
import numpy as np

import os

import time
import random
import re
import pandas as pd
import torch
from tqdm import tqdm
import sys
import seaborn as sns
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

sys.path.append("..")
# from data.mvtec import ADDataset
#from helper.summary import caculate_accuracy_mmad
from GPT4.gpt4v import GPT4Query, instruction

sys.path.append("../")

class QwenQuery(GPT4Query):
    def __init__(self, image_path, text_gt, tokenizer, model, few_shot=None,
                 visualization=False, args=None, log_path=None):
        super(QwenQuery, self).__init__(image_path, text_gt, few_shot or [], visualization)
        self.tokenizer = tokenizer
        self.model = model
        self.args = args
        self.log_path = log_path  # 可选：把 CoT/回答写入日志

    def _write_log(self, text: str):
        if not self.log_path:
            return
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def _build_payload(self, question_text: str):
        """构造 Qwen-VL list_format payload：few-shot图 + query图 + 问题"""
        incontext = []
        if self.few_shot:
            incontext.append({
                "text": f"Following are {len(self.few_shot)} reference normal images for comparison."
            })
            for ref_image_path in self.few_shot:
                if self.visualization:
                    ref_image = cv2.imread(ref_image_path)
                    self.visualize_image(ref_image)
                incontext.append({"image": ref_image_path})

        if self.visualization:
            img = cv2.imread(self.image_path)
            self.visualize_image(img)

        payload = (
            [{"text": instruction}]
            + incontext
            + [{"text": "Following is the query image:"}, {"image": self.image_path}]
            + [{"text": "Following is the question. Please answer in natural language based on the image."}]
            + [{"text": question_text}]
        )
        return payload

    def generate_answer(self):
        questions, answers = self.parse_conversation(self.text_gt)
        if questions == [] or answers == []:
            return questions, answers, None

        gpt_answers = []
        history = None

        #coi
        if self.args and getattr(self.args, "CoT", False):
            cot_prompt = (
                "Let's carefully analyze the reference and test images step by step:\n"
                "Step 1: The template image follows certain visual rules or regular patterns. Observe the template carefully and summarize these rules or expected visual characteristics.\n"
                "Step 2: Based on these rules, describe the differences between the template image and the test image, focusing on the left, right, and central regions.\n"
                "Step 3: From the differences identified in Step 2, determine which ones violate the rules in Step 1 and list them as inconsistencies or anomalies. Briefly explain why each one is anomalous.\n"
                "Step 4: Based on the anomalies listed in Step 3, make a final judgment on whether the test image contains any defects. Start with 'yes' if there is any anomaly in the test image, otherwise start with 'no'. If 'yes', describe the main anomaly and its location.\n"
            )
            cot_payload = self._build_payload(cot_prompt)
            cot_query = self.tokenizer.from_list_format(cot_payload)

            cot_response, cot_history = self.model.chat(self.tokenizer, query=cot_query, history=None)
            print(cot_response)

            self._write_log(f"\n=== CoT for {self.image_path} ===\n{cot_response}\n" + "-" * 80)

            # CoT 开了的话，默认把 history 接到后续问答里（即使 record_history=False，也能用上 CoT 的上下文）
            history = cot_history

        # ===== 正式回答问题（通常每张图只有一个问题，但保留循环）=====
        for i in range(len(questions)):
            part_questions = questions[i:i + 1]
            q_text = part_questions[0]["text"] if isinstance(part_questions[0], dict) else str(part_questions[0])

            payload = self._build_payload(q_text)
            query = self.tokenizer.from_list_format(payload)

            # 如果 record_history=True，就持续累积；否则只用当前 history（如果 CoT 开了）跑一次
            if self.args and self.args.record_history:
                response, history = self.model.chat(self.tokenizer, query=query, history=history)
            else:
                response, _ = self.model.chat(self.tokenizer, query=query, history=history)

            print(response)
            self._write_log(f"\n=== IMAGE: {self.image_path} ===\n{response}\n" + "-" * 80)

            # 自由文本：直接保存 response
            gpt_answers.append(response)

        return questions, answers, gpt_answers


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen-VL-Chat")
    parser.add_argument("--few_shot_model", type=int, default=1)
    parser.add_argument("--CoT", action="store_true")

    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--similar_template", action="store_true")
    parser.add_argument("--record_history", action="store_true")

    args = parser.parse_args()

    torch.manual_seed(1234)
    model_path = args.model_path
    # Note: The default behavior now has injection attack prevention off.
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir="~/.cache/huggingface/hub")

    # use cuda device
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, cache_dir="~/.cache/huggingface/hub").eval()
    # Specify hyperparameters for generation
    model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True, cache_dir="~/.cache/huggingface/hub")

    model_name = os.path.split(model_path.rstrip('/'))[-1]
    if args.similar_template:
        model_name = model_name + "_Similar_template"

    answers_json_path = f"result/answers_{args.few_shot_model}_shot_{model_name}.json"
    if not os.path.exists("result"):
        os.makedirs("result")
    print(f"Answers will be saved at {answers_json_path}")
    # For storing all answers
    if os.path.exists(answers_json_path):
        with open(answers_json_path, "r") as file:
            all_answers_json = json.load(file)
    else:
        all_answers_json = []

    existing_images = [a["image"] for a in all_answers_json]

    cfg = {
        "data_path": "../dataset",
        "json_path": "../dataset/breakfast_box/logical.json"
    }
    args.data_path = cfg["data_path"]

    with open(cfg["json_path"], "r") as file:
        chat_ad = json.load(file)

    for image_path in tqdm(chat_ad.keys()):
        if image_path in existing_images and not args.reproduce:
            continue
        text_gt = chat_ad[image_path]
        if args.similar_template:
            few_shot = text_gt["similar_templates"][:args.few_shot_model]
        else:
            few_shot = text_gt["random_templates"][:args.few_shot_model]

        rel_image_path = os.path.join(args.data_path, image_path)
        rel_few_shot = [os.path.join(args.data_path, path) for path in few_shot]
        log_path = "../result/qwen_log.txt"
        qwenquery = QwenQuery(image_path=rel_image_path, text_gt=text_gt,
                           tokenizer=tokenizer, model=model, few_shot=rel_few_shot, visualization=False, args=args,log_path=log_path)

        questions, answers, gpt_answers = qwenquery.generate_answer()
        if gpt_answers is None or len(gpt_answers) != len(answers):
            print(f"Error at {image_path}")
            continue
    
        # Update answer record
        for q, a, ga in zip(questions, answers, gpt_answers):
            answer_entry = {
                "image": image_path,
                "question": q,
                "gpt_answers": ga,
                "true_anwser": a
            }

            all_answers_json.append(answer_entry)

        # Save answers as JSON
        with open(answers_json_path, "w") as file:
            json.dump(all_answers_json, file, indent=4)

