import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_ROOT = "data/raw/"
PROCESSED_ROOT = "data/processed/"

LANGUAGE_DICT = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "it": "italian",
    "pt": "portuguese",
    "ro": "romanian",
}

os.makedirs(PROCESSED_ROOT, exist_ok=True)

directory_names = next(os.walk(RAW_ROOT))[1]

data = pd.DataFrame(columns=["source_text", "target_text", "source_language", "target_language"])

for directory_name in directory_names:
    print(f"Processing {directory_name}")
    target_language, source_language = directory_name.split("-")

    source_file = os.path.join(RAW_ROOT, directory_name, f"OpenSubtitles.{target_language}-{source_language}.{source_language}")
    target_file = os.path.join(RAW_ROOT, directory_name, f"OpenSubtitles.{target_language}-{source_language}.{target_language}")

    source_text = [line.strip() for line in open(source_file, "r").readlines()]
    target_text = [line.strip() for line in open(target_file, "r").readlines()]

    language_data = pd.DataFrame({
        "source_text": source_text,
        "target_text": target_text,
        "source_language": LANGUAGE_DICT[source_language],
        "target_language": LANGUAGE_DICT[target_language]
    }).head(10000)

    data = pd.concat([data, language_data])

print("Generating splits")

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data: pd.DataFrame = train_data.reset_index(drop=True)
test_data: pd.DataFrame = test_data.reset_index(drop=True)

print(f"Saving data to {os.path.join(PROCESSED_ROOT, 'train.csv')}")

train_data.to_csv(os.path.join(PROCESSED_ROOT, "train.csv"), index=False)

print(f"Saving data to {os.path.join(PROCESSED_ROOT, 'test.csv')}")

test_data.to_csv(os.path.join(PROCESSED_ROOT, "test.csv"), index=False)