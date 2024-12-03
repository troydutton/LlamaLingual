import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_name: str = "meta-llama/Llama-2-7b-chat-hf") -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads a HuggingFace model and tokenizer.
    """

    # Load in Bits & Bytes configuration
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load the pretrained model & tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        quantization_config=config,
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set the padding token to the eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    tokenizer.padding_side = "right"

    return model, tokenizer

def load_lora_model(adapter_name: str, model_name: str = "meta-llama/Llama-2-7b-chat-hf") -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads a pretrained model and tokenizer, then applies an adapter to the model.
    """

    # Load in Bits & Bytes configuration
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load the pretrained model & tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        quantization_config=config,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the adapter
    model = PeftModel.from_pretrained(model, adapter_name)

    # Set the padding token to the eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer
