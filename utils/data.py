from typing import Tuple

import pandas as pd
import torch
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
            truncation=True,
        )

        target_tokens = self.tokenizer(
            target_chat,
            max_length=self.max_length,
            truncation=True,
        )

        return {"input_ids": source_tokens["input_ids"], "labels": target_tokens["input_ids"]}



def collate(batch, tokenizer: AutoTokenizer) -> dict[str, Tensor]:
    max_length = max([max(len(element["input_ids"]), len(element["labels"])) for element in batch])

    input_ids, attention_masks, labels = [], [], []

    for element in batch:
        input_ids.append(element["input_ids"] + [tokenizer.pad_token_id] * (max_length - len(element["input_ids"])))
        attention_masks.append([1] * len(element["input_ids"]) + [0] * (max_length - len(element["input_ids"])))
        labels.append(element["labels"] + [-100] * (max_length - len(element["labels"])))

    batch = {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_masks)
    }

    return batch
