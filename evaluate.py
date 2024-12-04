import numpy as np
import pandas as pd
import sacrebleu
import torch
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.model import load_lora_model, load_model


def compute_translation_metrics(df):
# Initialize metrics storage
    metrics_data = {
        'sacrebleu_score': [],
        'chrf_score': [],
        'rouge_1_precision': [],
        'rouge_1_recall': [],
        'rouge_1_f1': [],
        'rouge_2_precision': [],
        'rouge_2_recall': [],
        'rouge_2_f1': [],
        'length_ratio': [],
        'word_error_rate': []
    }
    
    # Initialize ROUGE scorer
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    
    # Compute metrics for each row
    for _, row in df.iterrows():
        reference = row['target_text']
        hypothesis = row['translated_text']
        
        # Tokenize texts (simple whitespace tokenization)
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        # 1. SacreBLEU Score
        sacrebleu_score = sacrebleu.sentence_bleu(hypothesis, [reference]).score
        metrics_data['sacrebleu_score'].append(sacrebleu_score)
        
        # 2. chrF++ Score
        chrf_score = sacrebleu.sentence_chrf(hypothesis, [reference])
        metrics_data['chrf_score'].append(chrf_score.score)
        
        # 3. ROUGE Scores
        rouge_scores = rouge_scorer_obj.score(reference, hypothesis)
        metrics_data['rouge_1_precision'].append(rouge_scores['rouge1'].precision)
        metrics_data['rouge_1_recall'].append(rouge_scores['rouge1'].recall)
        metrics_data['rouge_1_f1'].append(rouge_scores['rouge1'].fmeasure)
        metrics_data['rouge_2_precision'].append(rouge_scores['rouge2'].precision)
        metrics_data['rouge_2_recall'].append(rouge_scores['rouge2'].recall)
        metrics_data['rouge_2_f1'].append(rouge_scores['rouge2'].fmeasure)
        
        # 4. Length Ratio
        metrics_data['length_ratio'].append(len(hyp_tokens) / len(ref_tokens) if len(ref_tokens) > 0 else 1.0)
        
        # 5. Word Error Rate (custom implementation)
        wer = word_error_rate(ref_tokens, hyp_tokens)
        metrics_data['word_error_rate'].append(wer)
    
    # Add metrics to DataFrame
    for metric_name, metric_values in metrics_data.items():
        df[metric_name] = metric_values
    
    return df

def word_error_rate(reference, hypothesis):
    """
    Compute Word Error Rate (WER)
    """
    # Compute Levenshtein distance
    def levenshtein_distance(s1, s2):
        m, n = len(s1), len(s2)
        # Create a matrix to store results of subproblems
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill d[][] in bottom up manner
        for i in range(m + 1):
            for j in range(n + 1):
                # If first string is empty, only option is to insert all characters of second string
                if i == 0:
                    dp[i][j] = j
                # If second string is empty, only option is to remove all characters of first string
                elif j == 0:
                    dp[i][j] = i
                # If last characters are same, ignore last char and recur for remaining string
                elif s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                # If last characters are different, consider all possibilities and find minimum
                else:
                    dp[i][j] = 1 + min(dp[i][j-1],      # Insert
                                       dp[i-1][j],      # Remove
                                       dp[i-1][j-1])    # Replace
        return dp[m][n]
    
    # Compute WER
    edits = levenshtein_distance(reference, hypothesis)
    wer = edits / len(reference) if len(reference) > 0 else 0
    return wer

def print_overall_metrics(df):
    """
    Print overall translation quality metrics
    """
    metric_columns = [
        'sacrebleu_score', 'chrf_score', 
        'rouge_1_f1', 'rouge_2_f1', 
        'length_ratio', 'word_error_rate'
    ]
    
    print("Overall Translation Quality Metrics:")
    print("-" * 40)
    
    for metric in metric_columns:
        try:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            print(f"{metric}:")
            print(f"  Mean: {mean_val:.4f}")
            print(f"  Std Dev: {std_val:.4f}")
        except Exception as e:
            print(f"Error computing {metric}: {e}")
    
    print("-" * 40)

def translate_text(model, tokenizer, data):
    def translate_single_row(row):
        source_text = row['source_text']
        source_language = row['source_language']
        target_language = row['target_language']

        # Prepare chat template
        chat = [
            {
                "role": "system",
                "content": f"Translate the following text from {source_language} to {target_language}:\n",
            },
            {"role": "user", "content": source_text},
        ]

        # Tokenize and generate translation
        chat = tokenizer.apply_chat_template(chat, tokenize=False)
        tokens = tokenizer(chat, return_tensors="pt").to(model.device)
        
        # Use generation settings for better quality and efficiency
        output_tokens = model.generate(
            **tokens, 
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        # Decode the output tokens
        output_text = tokenizer.decode(
            output_tokens[0, len(tokens["input_ids"][0]):], 
            skip_special_tokens=True
        )

        return output_text

    # Use pandas apply with a progress bar
    tqdm.pandas(desc="Translating")
    data['translated_text'] = data.progress_apply(translate_single_row, axis=1)

    return data

def main():
    CHECKPOINT_PATH = "checkpoints/checkpoint-40000"
    NUM_DATA_POINTS = 5000
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load evaluation dataset
    test_data = pd.read_csv("data/processed/test.csv").head(NUM_DATA_POINTS)

    # Get model and tokenizer
    model, tokenizer = load_lora_model(CHECKPOINT_PATH)
    
    # Move model to appropriate device
    model = model.to(device)
    model.eval()

    # Translate and evaluate all source sentences
    with torch.no_grad():  # Disable gradient computation for inference
        translated_data = translate_text(model, tokenizer, test_data)

    # Save translated sentences with BLEU scores
    translated_data.to_csv(f"data/processed/test_translated_{CHECKPOINT_PATH.split('/')[1]}.csv", index=False)
    
    # translated_data = pd.read_csv("data/processed/test_translated_checkpoint-10000.csv")

    # Compute metrics
    evaluated_df = compute_translation_metrics(translated_data)

    # Print overall metrics
    print_overall_metrics(evaluated_df)
    
    # Save to CSV with metrics
    evaluated_df.to_csv(f"data/processed/translation_metrics_{CHECKPOINT_PATH.split('/')[1]}.csv", index=False)

if __name__ == "__main__":
    main()