# LlamaLingual

LlamaLingual is a Llama 2 model fine-tuned to translate from the Romance Languages to English.

The methodology and results can be found in the [report](resources/report.pdf).

## Setup

1. Install the required dependencies:
```bash
mamba env create -f environment.yaml
```

2. Activate the environment:
```bash
conda activate llama
```

3. Install bitsandbytes package from source to enable quantization:
```bash
git clone https://github.com/TimDettmers/bitsandbytes.git && cd bitsandbytes/
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=cuda -S .
make
pip install .
```

4. Log in to Hugging Face 🤗:
```bash
huggingface-cli login
```

5. Log in to W&B:
```bash
wandb login
```

6. Download the data:
```bash
bash download_data.sh
```

7. Preprocess the data:
```bash
python preprocess.py
```

## Usage

Fine-tune the model:
```bash
python finetune.py
```

## Dataset
The data used for fine-tuning is the [OpenSubtitles dataset](https://opus.nlpl.eu/OpenSubtitles/corpus/version/OpenSubtitles).