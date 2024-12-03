import pandas as pd
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import Trainer, TrainingArguments

from utils.data import TranslationDataset, tokenize_sample
from utils.model import load_model

tqdm.pandas()

def main() -> None:
    # Load the model & tokenizer
    model, tokenizer = load_model()
    
    # Load in the data
    print("Loading data")

    train_data = pd.read_csv("data/processed/train.csv")
    eval_data = pd.read_csv("data/processed/test.csv")

    # Tokenize the data
    train_data["source_tokens"], train_data["target_tokens"] = zip(*train_data.progress_apply(lambda row: tokenize_sample(tokenizer, row), axis=1))
    eval_data["source_tokens"], eval_data["target_tokens"] = zip(*eval_data.progress_apply(lambda row: tokenize_sample(tokenizer, row), axis=1))

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

    # Setup trainer
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        run_name="Llama-2",
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        learning_rate=1e-4,
        num_train_epochs=2,
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=20000,
        save_steps=5000,
        report_to="wandb",
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=TranslationDataset(train_data),
        eval_dataset=TranslationDataset(eval_data),
    )

    # # Now train!
    trainer.train()

if __name__ == "__main__":
    main()
