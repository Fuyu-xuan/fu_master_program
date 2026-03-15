import argparse

import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

import pandas as pd
import torch
from tqdm import tqdm
import sys

import seaborn as sns
sys.path.append("..")

from helper.summary import caculate_accuracy_mmad
from GPT4.gpt4v import GPT4Query, instruction
from SoftPatch.call import call_patchcore, build_patchcore


from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path

class LLaVAQuery(GPT4Query):
    def __init__(self, image_path, text_gt, tokenizer, model, image_processor, context_len, few_shot=[], defect_shot=[], visualization=False, args=None):
        super(LLaVAQuery, self).__init__(image_path, text_gt, few_shot, visualization)
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len
        self.defect_shot = defect_shot
        self.args = args

    def generate_answer(self):

        log_path = "result/llava_cot_log.txt"

        questions, answers = self.parse_conversation(self.text_gt)
        if questions == [] or answers == []:
            return questions, answers, None

        gpt_answers = []
        history = []  # [(q,a),...]

        # ===== CoT (optional) =====
        if self.args and self.args.CoT:
            cot_prompt = [{
                "text":
                    "Let's carefully analyze the template and test images step by step:\n"
                    "Step 1:The template image follows certain visual rules or regular patterns. Observe the template carefully and summarize these rules or expected visual characteristics.\n"
                    "Step 2:Based on these rules, describe the differences between the template image and the test image, focusing on the left, right, and central regions.\n"
                    "Step 3: From the differences identified in Step 2, determine which ones violate the rules in Step 1 and list them as inconsistencies or anomalies. Briefly explain why each one is anomalous.\n"
                    "Step 4: Based on the anomalies listed in Step 3, make a final judgment on whether the test image contains any defects. Start with 'yes' if there is any anomaly in the test image, otherwise start with 'no'. If 'yes', describe the main anomaly and its location.\n"
            }]

            input_ids, image_tensor, image_sizes = self.get_query(cot_prompt, history)
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if self.args.temperature > 0 else False,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    num_beams=self.args.num_beams,
                    max_new_tokens=512,
                    use_cache=True,
                )
            cot_out = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print(cot_out)
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n=== CoT for {self.image_path} ===\n{cot_out}\n" + "-" * 80 + "\n")
            history.append((cot_prompt[0]["text"], cot_out))

        # ===== real question(s) =====
        for i in range(len(questions)):
            part_questions = questions[i:i+1]
            input_ids, image_tensor, image_sizes = self.get_query(part_questions, history)
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if self.args.temperature > 0 else False,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    num_beams=self.args.num_beams,
                    max_new_tokens=128,
                    use_cache=True,
                )
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print(outputs)

            if self.args and self.args.record_history:
                history.append((part_questions[0]["text"], outputs))

            gpt_answers.append(outputs)

        return questions, answers, gpt_answers



    def parse_conversation(self, text_gt):
        questions = []
        answers = []


        for key, conv in text_gt.items():
            if not key.startswith("conversation"):
                continue

            for qa in conv:

                q_text = qa.get("Question") or qa.get("question") or qa.get("Q") or ""
                a_text = qa.get("Answer") or qa.get("answer") or qa.get("A") or ""
                if q_text == "" and isinstance(qa, dict) and "text" in qa:
                    q_text = qa["text"]

                questions.append({"text": q_text})
                answers.append(a_text)
            break

        return questions, answers


    def get_query(self, conversations, history=None):
        history = history or []
        hint = instruction

        prompt_txt = ""

        # template images (few_shot)
        if self.few_shot:
            prompt_txt += f"Following are {len(self.few_shot)} reference normal images for comparison:\n"
            prompt_txt += (DEFAULT_IMAGE_TOKEN + "\n") * len(self.few_shot)

        # query image
        prompt_txt += "Following is the query image:\n"
        prompt_txt += DEFAULT_IMAGE_TOKEN + "\n\n"

        # history
        if history:
            prompt_txt += "Previous Q&A:\n"
            for q, a in history:
                prompt_txt += f"Q: {q}\nA: {a}\n"

        # 单问题
        question_text = conversations[0]["text"] if conversations else ""
        prompt_txt += "Following is the question. Please answer in natural language based on the image.\n"
        prompt_txt += question_text + "\n"

        qs = hint + "\n" + prompt_txt

        conv = conv_templates[self.args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        # ===== images: template + defect + query（顺序必须和 token 数量匹配）=====
        query_image = cv2.imread(self.image_path)
        if self.visualization:
            self.visualize_image(query_image)
        query_image = load_image_from_base64(self.encode_image_to_base64(query_image))

        ref_images = []
        for ref_image_path in self.few_shot:
            img = cv2.imread(ref_image_path)
            if self.visualization:
                self.visualize_image(img)
            ref_images.append(load_image_from_base64(self.encode_image_to_base64(img)))

        for ref_image_path in self.defect_shot:
            img = cv2.imread(ref_image_path)
            if self.visualization:
                self.visualize_image(img)
            ref_images.append(load_image_from_base64(self.encode_image_to_base64(img)))

        images = ref_images + [query_image]

        if hasattr(self.image_processor, "forward"):
            image_tensor = process_images(images, self.image_processor, self.context_len)
        else:
            image_tensor = self.image_processor.preprocess(images, return_tensors="pt")["pixel_values"]

        image_tensor = [image_tensor[i].unsqueeze(0).half().cuda() for i in range(len(image_tensor))]
        image_sizes = [img.size for img in images]

        if self.args.text_only:
            return input_ids, None, None

        return input_ids, image_tensor, image_sizes

 



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")

    parser.add_argument("--dtype", type=str, default="fp32")

    parser.add_argument("--few_shot_model", type=int, default=1)
    parser.add_argument("--similar_template", action="store_true")
    parser.add_argument("--reproduce", action="store_true")

    # parser.add_argument("--defect-shot", type=int, default=1)
    parser.add_argument("--record_history", action="store_true")
    parser.add_argument("--CoT", action="store_true")


    args = parser.parse_args()

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    if args.dtype == "4bit":
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, load_4bit=True)
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    model_name = os.path.split(model_path.rstrip('/'))[-1]
    if args.CoT:
        model_name = model_name + "_CoT"

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
        llavaquery = LLaVAQuery(image_path=rel_image_path, text_gt=text_gt,
                           tokenizer=tokenizer, model=model, image_processor=image_processor, context_len=context_len,
                           few_shot=rel_few_shot, visualization=False, args=args)
        questions, answers, gpt_answers = llavaquery.generate_answer()
        if gpt_answers is None or len(gpt_answers) != len(answers):
            print(f"Error at {image_path}")
            continue

        questions_type = [conversion["type"] for conversion in text_gt["conversation"]]
        # Update answer record
        for q, a, ga in zip(questions, answers, gpt_answers):
            answer_entry = {
                "image": image_path,
                "question": q,
                "correct_answer": a,
                "gpt_answer": ga
            }

            all_answers_json.append(answer_entry)

        # Save answers as JSON
        with open(answers_json_path, "w") as file:
            json.dump(all_answers_json, file, indent=4)

