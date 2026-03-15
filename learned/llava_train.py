import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor,
    Trainer,
    TrainingArguments,
)

from train_llava.data import LlavaDataset, TrainLLavaModelCollator
from train_llava.util import print_trainable_parameters

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    base_model_name_or_path: Optional[str] = field(
        default="llava-hf/llava-1.5-7b-hf",
        metadata={"help": "Path to the base model"}
    )
    train_type: Optional[str] = field(
        default="use_lora",
        metadata={
            "help": """
            1. use_lora: 使用LoRA训练
            2. none: 全量参数训练
            3. freeze_vision: 冻结vision tower，只训练其余部分
            """
        },
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None,
        metadata={"help": "Path to the training data."}
    )


def load_model_processor(model_args: ModelArguments):
    model = LlavaForConditionalGeneration.from_pretrained(
        model_args.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    processor = LlavaProcessor.from_pretrained(
        model_args.base_model_name_or_path
    )

    if model_args.train_type == "use_lora":
        print("使用 LoRA 训练")

        lora_config = LoraConfig(
            r=32,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["multi_modal_projector"],
        )

        model = get_peft_model(model, lora_config)

    elif model_args.train_type == "none":
        logging.warning("使用全量参数进行训练")

    elif model_args.train_type == "freeze_vision":
        logging.warning("冻结 vision tower，仅训练其余部分")

        # LLaVA-1.5 在 HF 结构里通常是 model.model.vision_tower
        for param in model.model.vision_tower.parameters():
            param.requires_grad = False

    else:
        raise ValueError(f"Unknown train_type: {model_args.train_type}")

    print_trainable_parameters(model)
    return model, processor


def load_dataset_collator(processor, data_args: DataArguments):
    train_dataset = LlavaDataset(data_args.data_path)
    data_collator = TrainLLavaModelCollator(processor, -100)
    return train_dataset, data_collator


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model, processor = load_model_processor(model_args)
    train_dataset, data_collator = load_dataset_collator(processor, data_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()