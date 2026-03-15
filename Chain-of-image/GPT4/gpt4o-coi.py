import argparse
import base64
import json
import requests
import cv2
import matplotlib.pyplot as plt
import os
import time
import random
import re
import pandas as pd
from requests import RequestException
from tqdm import tqdm
import sys
from difflib import get_close_matches

sys.path.append("..")
from helper.summary import caculate_accuracy_mmad

error_keywords = ['please', 'sorry', 'today', 'cannot assist']
api = {
    "api_key": "your_key",  # 请替换
    "url": "https://api.openai.com/v1/chat/completions"
}

# -------- GPT4Query 基类 ----------
class GPT4Query():
    def __init__(self, image_path, text_gt, few_shot=[], visualization=False, CoT=False):
        self.api_key = api["api_key"]
        self.url = api["url"]
        self.image_path = image_path
        self.text_gt = text_gt
        self.few_shot = few_shot
        self.max_image_size = (512, 512)
        self.api_time_cost = 0
        self.visualization = visualization
        self.max_retries = 5
        self.CoT = CoT  # 是否启用Chain-of-Thought

    # ---------- 图像与可视化 ----------
    def encode_image_to_base64(self, image):
        height, width = image.shape[:2]
        scale = min(self.max_image_size[0] / width, self.max_image_size[1] / height)
        new_width, new_height = int(width * scale), int(height * scale)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        _, encoded_image = cv2.imencode('.jpg', resized_image)
        return base64.b64encode(encoded_image).decode('utf-8')

    def visualize_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.show()

    # ---------- API 请求 ----------
    def send_request_to_api(self, payload):
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        retries, retry_delay = 0, 1
        while retries < self.max_retries:
            try:
                before = time.time()
                response = requests.post(self.url, headers=headers, json=payload)
                choices = response.json().get('choices', [])
                if choices:
                    content = choices[0]['message']['content']
                    if any(word in content.lower() for word in error_keywords):
                        print(f"Error respond of {self.image_path}: {content}")
                        retries += 1
                        continue
                    self.api_time_cost += time.time() - before
                    return response.json()
                else:
                    print(response.json())
                    retries += 1
            except RequestException as e:
                print(f"Request failed: {e}, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
                retries += 1
        print("Failed to send request.")
        return None

    # ---------- 数据解析 ----------
    def parse_conversation(self, text_gt):
        Question, Answer = [], []
        keyword = "conversation"
        for key in text_gt.keys():
            if key.startswith(keyword):
                conversation = text_gt[key]
                for i, QA in enumerate(conversation):
                    options_items = list(QA['Options'].items())
                    options_text = ""
                    for j, (key_opt, value) in enumerate(options_items):
                        options_text += f"{key_opt}. {value}\n"
                    questions_text = QA['Question']
                    Question.append({"type": "text",
                                     "text": f"Question {i + 1}: {questions_text}\n{options_text}"})
                    Answer.append(QA['Answer'])
                break
        return Question, Answer

    def parse_answer(self, response_text, options=None):
        pattern = re.compile(r'\b([A-E])\b')
        answers = pattern.findall(response_text)
        if not answers and options:
            options_values = list(options.values())
            closest = get_close_matches(response_text, options_values, n=1, cutoff=0.0)
            if closest:
                for key, value in options.items():
                    if value == closest[0]:
                        answers.append(key)
                        break
        return answers

    def parse_json(self, response_json):
        choices = response_json.get('choices', [])
        if choices:
            caption = choices[0].get('message', {}).get('content', '')
            print("\n========== [GPT-4o OUTPUT] ==========")
            print(caption)
            print("=====================================\n")
            return caption
            #if self.visualization:
            #   print(f"Caption: {caption}")
            #return caption
        return ''

    # ---------- 构造输入 ----------
    def get_query(self, conversation, is_cot=False):
        if self.CoT and is_cot:
            instruction = """
You are an industrial inspector who checks products by images.
First output 'yes' or 'no' (the first token must be exactly yes/no).
If 'yes', describe the main anomaly and its location.
"""
        else:
            instruction = '''
You are an industrial inspector who checks products by images.
You should judge whether there is a defect in the query image and answer the questions about it.
'''

        incontext = []
        if self.few_shot:
            incontext.append({"type": "text",
                              "text": f"Following is {len(self.few_shot)} image(s) of normal samples for comparison."})
            for ref_image_path in self.few_shot:
                ref_image = cv2.imread(ref_image_path)
                ref_base64_image = self.encode_image_to_base64(ref_image)
                incontext.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{ref_base64_image}", "detail": "low"}
                })

        image = cv2.imread(self.image_path)
        base64_image = self.encode_image_to_base64(image)

        conversation_text = ''.join([f"{q['text']}\n" for q in conversation])

        payload = {
            "model": "gpt-4o",
            "messages": [{
                "role": "user",
                "content": incontext + [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}},
                    {"type": "text",
                     "text": instruction + '\n' + "Following is the question list:\n" + conversation_text}
                ]
            }],
            "max_tokens": 400,
        }
        return payload

    # ---------- 主推理过程 ----------
    def generate_answer(self,log_path=None):
        questions, answers = self.parse_conversation(self.text_gt)
        if not questions or not answers:
            return questions, answers, None, None

        gpt_answers = []
        reasoning_text = None

        # --- COT 阶段 ---
        if self.CoT:
            reasoning_prompt = [{"type": "text",
                                 "text": ( "Let's carefully analyze the template and test images step by step:\n"
                                            "Step 1: The template image follows certain visual rules or regular patterns. Observe the template carefully and summarize these rules or expected visual characteristics.\n"
                                            "Step 2: Based on these rules, describe the differences between the template image and the test image, focusing on the left, right, and central regions.\n"
                                            "Step 3: From the differences identified in Step 2, determine which ones violate the rules in Step 1 and list them as inconsistencies or anomalies. Briefly explain why each one is anomalous.\n"
                                            "Step 4: Based on the anomalies listed in Step 3, make a final judgment on whether the test image contains any defects. ")}]
            payload = self.get_query(reasoning_prompt, is_cot=True)
            reasoning_response = self.send_request_to_api(payload)
            if reasoning_response:
                reasoning_text = self.parse_json(reasoning_response)
                print(f"[CoT reasoning]\n{reasoning_text}\n")
                # --- 写入日志 ---
                if log_path:
                    write_log(log_path, f"\n[{self.image_path}]")
                    write_log(log_path, "[CoT reasoning process]:")
                    write_log(log_path, reasoning_text)
                    write_log(log_path, "-" * 60)
        # --- 回答阶段 ---
        for i in range(len(questions)):
            part_questions = questions[i:i+1]  # 单问题
            payload = self.get_query(part_questions)
            respond = self.send_request_to_api(payload)
            if respond is None:
                gpt_answers.append('')
                continue
            resp_text = self.parse_json(respond)   # 原样文本
            gpt_answers.append(resp_text)

        return questions, answers, gpt_answers, reasoning_text

def write_log(log_path, text):
    """简单日志追加写入函数"""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# ---------- 主程序 ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--few_shot_model", type=int, default=1)
    parser.add_argument("--CoT", action="store_true")
    args = parser.parse_args()
    
    model_name = "gpt-4o"
    if args.CoT:
        model_name += "_CoT"

    answers_json_path = f"result/answers_{args.few_shot_model}_shot_{model_name}.json"
    if not os.path.exists("result"):
        os.makedirs("result")

    if os.path.exists(answers_json_path):
        with open(answers_json_path, "r") as file:
            all_answers_json = json.load(file)
    else:
        all_answers_json = []
    log_path = os.path.join("result", f"log_{args.few_shot_model}_shot_{model_name}.txt")
    if not os.path.exists("result"):
        os.makedirs("result")
    cfg = {
        "data_path": "../dataset",
        "json_path": "../dataset/breakfast_box/logical.json"
    }
    args.data_path = cfg["data_path"]

    with open(cfg["json_path"], "r") as file:
        chat_ad = json.load(file)

    for data_id, image_path in enumerate(tqdm(chat_ad.keys())):
        text_gt = chat_ad[image_path]
        few_shot = text_gt["random_templates"][:args.few_shot_model]
        rel_image_path = os.path.join(args.data_path, image_path)
        rel_few_shot = [os.path.join(args.data_path, path) for path in few_shot]
        model = GPT4Query(image_path=rel_image_path, text_gt=text_gt, few_shot=rel_few_shot, CoT=args.CoT)
        questions, answers, gpt_answers, reasoning_text = model.generate_answer(log_path=log_path)
        if gpt_answers is None or len(gpt_answers) != len(answers):
            print(f"Error at {image_path}")
            continue

        write_log(log_path, f"\n[{image_path}]")
        if reasoning_text:
            write_log(log_path, f"CoT reasoning:\n{reasoning_text}\n")
        

        print(f"API time cost: {model.api_time_cost:.2f}s")

        for q, a, ga in zip(questions, answers, gpt_answers):
            answer_entry = {
                "image": image_path,
                "question": q,
                "correct_answer": a,
                "gpt_answer": ga,
                "cot_reasoning": reasoning_text if args.CoT else None
            }
            all_answers_json.append(answer_entry)

        with open(answers_json_path, "w") as file:
            json.dump(all_answers_json, file, indent=4)

