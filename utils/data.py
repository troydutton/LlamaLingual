from typing import Tuple

import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TranslationDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_tokens, target_tokens = self.data.iloc[idx][["source_tokens", "target_tokens"]]

        return {"input_ids": source_tokens, "labels": target_tokens}

def tokenize_sample(tokenizer: AutoTokenizer, sample: pd.Series, max_length: int = 512) -> Tuple[Tensor, Tensor]:
    """
    Tokenizes the source and target text the tokens.
    """

    # Tokenize the source text
    source_chat = [
        {"role": "system", "content": f"Translate the following text from {sample['source_language']} to {sample['target_language']}:\n"},
        {"role": "user", "content": sample["source_text"]}
    ]

    source_tokens = tokenizer.apply_chat_template(
        source_chat,
        padding="max_length",
        max_length=max_length,
        add_generation_prompt=True,
        return_tensors="pt",
    ).squeeze()

    target_chat = [
        {"role": "system", "content": f"Translate the following text from {sample['source_language']} to {sample['target_language']}:\n"},
        {"role": "user", "content": sample["source_text"]},
        {"role": "assistant", "content": sample["target_text"]}
    ]

    target_tokens = tokenizer.apply_chat_template(
        target_chat,
        padding="max_length",
        max_length=max_length,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    return source_tokens, target_tokens