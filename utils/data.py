from typing import Tuple

import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TranslationDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]

        return self.tokenize_sample(sample)

    def tokenize_sample(self, sample: pd.Series) -> dict:
        source_chat = [
            {"role": "system", "content": f"Translate the following text from {sample['source_language']} to {sample['target_language']}:\n"},
            {"role": "user", "content": sample["source_text"]}
        ]

        target_chat = [
            {"role": "system", "content": f"Translate the following text from {sample['source_language']} to {sample['target_language']}:\n"},
            {"role": "user", "content": sample["source_text"]},
            {"role": "assistant", "content": sample["target_text"]}
        ]

        source_chat = self.tokenizer.apply_chat_template(source_chat, tokenize=False)

        target_chat = self.tokenizer.apply_chat_template(target_chat, tokenize=False)

        source_tokens = self.tokenizer(
            source_chat,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target_tokens = self.tokenizer(
            target_chat,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {"input_ids": source_tokens["input_ids"].squeeze(), "attention_mask": source_tokens["attention_mask"].squeeze(), "labels": target_tokens["input_ids"].squeeze()}
