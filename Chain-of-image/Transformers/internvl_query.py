import argparse
import base64
import json
from collections import defaultdict

import math
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

import time
import random
import re
import pandas as pd
import torch
from tqdm import tqdm
import sys
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T

sys.path.append("..")
from helper.summary import caculate_accuracy_mmad
from GPT4.gpt4v import GPT4Query, instruction


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    #寻找最接近给定宽高比的图像尺寸。
    best_ratio_diff = float('inf') #初始化最佳宽高比
    best_ratio = (1, 1) 
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1] #计算当前目标的宽高比
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)  # 计算当前宽高比与目标宽高比的差异
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InternVLQuery(GPT4Query):
    def __init__(self, image_path, text_gt, tokenizer, model, few_shot=[], visualization=False, domain_knowledge=None, agent=None, mask_path=None, CoT=None, defect_shot=[], args=None):
        super(InternVLQuery, self).__init__(image_path, text_gt, few_shot, visualization)
        self.tokenizer = tokenizer
        self.model = model
        self.domain_knowledge = domain_knowledge
        self.agent = agent
        self.mask_path = mask_path
        self.CoT = CoT
        self.defect_shot = defect_shot
        self.args = args

    def generate_answer(self):
        log_path = "../result-cot/coi_log.txt"
        questions, answers = self.parse_conversation(self.text_gt)
        if questions == [] or answers == []:
            return questions, answers, None
        query_image = load_image(self.image_path, max_num=1).to(torch.bfloat16).cuda() # Default is 12 patches
        template_image = []
        for ref_image_path in self.few_shot:
            template_image.append(load_image(ref_image_path, max_num=1).to(torch.bfloat16).cuda())
        images = template_image + [query_image]
        pixel_values = torch.cat(images, dim=0)

        num_patches_list = [image.shape[0] for image in images]

        gpt_answers = []
        history = None
        #chain-of-image手法
        if self.CoT:
            chain_of_thought = []
            chain_of_thought.append(
                {
                    # "type": "text",
                    "text": f"Let's carefully analyze the template and test images step by step:\n" 
                            f"Step 1:The template image follows certain visual rules or regular patterns. Observe the template carefully and summarize these rules or expected visual characteristics.\n"
                            f"Step 2: Based on these rules, describe the differences between the template image and the test image, focusing on the left, right, and central regions.\n"
                            f"Step 3: From the differences identified in Step 2, determine which ones violate the rules in Step 1 and list them as inconsistencies or anomalies. Briefly explain why each one is anomalous.\n"
                            f"Step 4: Based on the anomalies listed in Step 3, make a final judgment on whether the test image contains any defects.Start with 'yes' if there is any anomaly in the test image, otherwise start with 'no'. If 'yes', describe the main anomaly and its location.\n"
                },
            )

            payload, conversation_text = self.get_query(chain_of_thought)
            query = payload + conversation_text
            response, temp_history = self.model.chat(tokenizer, pixel_values, query,
                                                dict(max_new_tokens=256, do_sample=False),
                                                num_patches_list=num_patches_list,
                                                history=history, return_history=True)
            print(response)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n=== CoT for {self.image_path} ===\n")
                f.write(response)
                f.write("\n" + "-"*80 + "\n")
            history = temp_history

        for i in range(len(questions)):
            part_questions = questions[i:i + 1]
            payload, conversation_text = self.get_query(part_questions)
            query = payload + conversation_text
            response, temp_history = self.model.chat(tokenizer, pixel_values, query,
                                                dict(max_new_tokens=128, do_sample=False),
                                                num_patches_list=num_patches_list,
                                                history=history, return_history=True)
            if args.record_history:
                history = temp_history
            print(response)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n=== IMAGE: {self.image_path} ===\n{response}\n" + "-"*80 + "\n")

            gpt_answer = self.parse_answer(response)
            gpt_answers.append(gpt_answer[-1])
        print(gpt_answers)
        return questions, answers, gpt_answers

    def get_query(self, conversation):
        incontext = ''
        if self.visualization:
            print(conversation)
        payload = (
            instruction + incontext + "\n"
                "Following is the query image:\n<image>\n"
                "Following is the question. Please answer in natural language based on the image.\n"
    )
    
        conversation_text = conversation[0]["text"] + "\n"
        return payload, conversation_text


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
         'InternVL2-8B': 32,}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="OpenGVLab/InternVL2-8B")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--few_shot_model", type=int, default=1) 

    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--similar_template", action="store_true")
    parser.add_argument("--record_history", action="store_true")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--visualization", action="store_true")
    parser.add_argument("--CoT", action="store_true")


    args = parser.parse_args()

    torch.manual_seed(1234)
    model_path = args.model_path
    model_name = os.path.split(model_path.rstrip('/'))[-1]

    torch.set_grad_enabled(False)
    if args.num_gpus > 1:
        device_map = split_model(model_name)
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map).eval()
    else:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    if args.similar_template:
        model_name += "_Similar_template"
    agent = None
    if args.CoT:
        model_name += "_CoT"
    if args.debug:
        model_name += "_Debug"


    answers_json_path = f"result/answers_{args.few_shot_model}_shot_{model_name}-1202.json"
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
        "json_path": "../dataset/breakfast_box/logical.json",
    }
    args.data_path = cfg["data_path"]

    with open(cfg["json_path"], "r") as file:
        chat_ad = json.load(file)

    if args.debug:
        # Fix random seed
        random.seed(1)
        # random.seed(10)
        sample_keys = random.sample(list(chat_ad.keys()), 1600)
    else:
        sample_keys = chat_ad.keys()


    for data_id, image_path in enumerate(tqdm(sample_keys)):
        if image_path in existing_images and not args.reproduce:
            continue
        text_gt = chat_ad[image_path]
        if args.similar_template:
            few_shot = text_gt["similar_templates"][:args.few_shot_model]
        else:
            few_shot = text_gt["random_templates"][:args.few_shot_model]

        rel_image_path = os.path.join(args.data_path, image_path)
        rel_few_shot = [os.path.join(args.data_path, path) for path in few_shot]

        internvlquery = InternVLQuery(image_path=rel_image_path, text_gt=text_gt,
                                      tokenizer=tokenizer, model=model, few_shot=rel_few_shot, visualization=args.visualization,
                                       CoT=args.CoT, args=args)
        questions, answers, gpt_answers = internvlquery.generate_answer()


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

        if data_id % 10 == 0 or data_id == len(chat_ad.keys()) - 1:
            # Save answers as JSON
            with open(answers_json_path, "w") as file:
                json.dump(all_answers_json, file, indent=4)

