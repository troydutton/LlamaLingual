import torchvision

torchvision.disable_beta_transforms_warning()

import logging

logging.getLogger("bitsandbytes").setLevel(logging.CRITICAL)

import os

os.environ["WANDB_PROJECT"] = "LlamaLingual"

import pandas as pd
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import ProgressCallback


def on_log(self, args, state, control, logs=None, **kwargs):
    if state.is_local_process_zero and self.training_bar is not None:
        _ = logs.pop("total_flos", None)

ProgressCallback.on_log = on_log

from utils.data import TranslationDataset, collate
from utils.model import load_model

tqdm.pandas()

def main() -> None:
    # Load the model & tokenizer
    model, tokenizer = load_model()
    
    # Add Lora Matrices
    lora_config = LoraConfig(
        r=20,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.4,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    peft_model = get_peft_model(model, lora_config)

    print(f"Number of trainable parameters: {sum(p.numel() for p in peft_model.parameters() if p.requires_grad)}")

    # Load the data
    train_data = pd.read_csv("data/processed/train.csv")
    eval_data = pd.read_csv("data/processed/test.csv")

    # Setup trainer
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        run_name="Llama-2",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=1e-4,
        num_train_epochs=8,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=20000,
        save_steps=10000,
        report_to="wandb"
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=TranslationDataset(train_data, tokenizer),
        eval_dataset=TranslationDataset(eval_data, tokenizer),
        data_collator=lambda batch: collate(batch, tokenizer),
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
