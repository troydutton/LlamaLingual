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
        chat = [
            {"role": "system", "content": f"Translate the following text from {sample['source_language']} to {sample['target_language']}:\n"},
            {"role": "user", "content": sample["source_text"]},
            {"role": "assistant", "content": sample["target_text"]}
        ] 

        chat = self.tokenizer.apply_chat_template(chat, tokenize=False)

        tokens = self.tokenizer(chat, max_length=self.max_length, truncation=True)

        return {"input_ids": tokens["input_ids"]}

def collate(batch, tokenizer: AutoTokenizer) -> dict[str, Tensor]:
    max_length = max([len(element["input_ids"]) for element in batch])

    batch_input_ids, batch_attention_masks, batch_labels = [], [], []

    for element in batch:
        pad_length = max_length - len(element["input_ids"])

        input_ids = element["input_ids"] + [tokenizer.pad_token_id] * pad_length
        attention_mask = [1] * len(element["input_ids"]) + [0] * pad_length
        labels = element["input_ids"] + [-100] * pad_length

        batch_input_ids.append(input_ids)
        batch_attention_masks.append(attention_mask)
        batch_labels.append(labels)

    batch = {
        "input_ids": torch.tensor(batch_input_ids),
        "labels": torch.tensor(batch_labels),
        "attention_mask": torch.tensor(batch_attention_masks)
    }

    return batch
