import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments

from train_internvl.data import InternVLDataset, TrainInternVLModelCollator
from train_internvl.utils import print_trainable_parameters

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="OpenGVLab/InternVL2-8B",
        metadata={"help": "Path to InternVL model"}
    )
    train_type: Optional[str] = field(
        default="use_lora",
        metadata={"help": "use_lora / none / freeze_vision"}
    )
    max_length: int = field(default=2048)
    use_flash_attn: bool = field(default=True)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None,
        metadata={"help": "Path to training json"}
    )


def load_model_tokenizer(model_args: ModelArguments):
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=model_args.use_flash_attn,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_args.train_type == "use_lora":
        print("Using LoRA training")

        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    elif model_args.train_type == "freeze_vision":
        print("Freeze vision encoder")
        if hasattr(model, "vision_model"):
            for p in model.vision_model.parameters():
                p.requires_grad = False

    elif model_args.train_type == "none":
        print("Full finetuning")

    else:
        raise ValueError(f"Unknown train_type: {model_args.train_type}")

    print_trainable_parameters(model)
    return model, tokenizer


def load_dataset_collator(tokenizer, model_args, data_args):
    train_dataset = InternVLDataset(data_args.data_path)
    data_collator = TrainInternVLModelCollator(
        tokenizer=tokenizer,
        image_size=448,
        max_length=model_args.max_length,
        ignore_index=-100,
    )
    return train_dataset, data_collator


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model, tokenizer = load_model_tokenizer(model_args)
    train_dataset, data_collator = load_dataset_collator(tokenizer, model_args, data_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()