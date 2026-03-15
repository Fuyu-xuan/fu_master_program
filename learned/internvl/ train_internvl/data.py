import json
import random
from typing import List, Dict

import torch
from PIL import Image
from torch.utils.data import Dataset


class InternVLDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image_path = item["image_paths"]
        image = Image.open(image_path).convert("RGB")

        conversation_sets = item["conversation_sets"]
        selected_conv = random.choice(conversation_sets)

        question = None
        answer = None

        for turn in selected_conv:
            if turn["from"] == "human":
                question = turn["value"]
            elif turn["from"] == "gpt":
                answer = turn["value"]

        if question is None or answer is None:
            raise ValueError(f"Invalid conversation format at index {idx}")

        return {
            "image": image,
            "question": question,
            "answer": answer,
        }


class TrainInternVLModelCollator:
    def __init__(self, tokenizer, image_size=448, max_length=2048, ignore_index=-100):
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_length = max_length
        self.ignore_index = ignore_index

        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def _build_text(self, question: str, answer: str):
        prompt = f"<image>\nUser: {question.strip()}\nAssistant:"
        full_text = f"{prompt} {answer.strip()}"
        return prompt, full_text

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        pixel_values_list = []

        for feat in features:
            image = feat["image"]
            question = feat["question"].replace("<image>", "").strip()
            answer = feat["answer"].strip()

            prompt_text, full_text = self._build_text(question, answer)

            prompt_ids = self.tokenizer(
                prompt_text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )["input_ids"][0]

            full_ids = self.tokenizer(
                full_text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )["input_ids"][0]

            labels = full_ids.clone()
            prompt_len = prompt_ids.shape[0]
            labels[:prompt_len] = self.ignore_index

            attention_mask = torch.ones_like(full_ids)
            pixel_values = self.transform(image)

            input_ids_list.append(full_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)
            pixel_values_list.append(pixel_values)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=self.ignore_index,
        )

        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask_list,
            batch_first=True,
            padding_value=0,
        )

        pixel_values = torch.stack(pixel_values_list, dim=0)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }