import json
import os

# Resolve oneDNN warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from evaluate import load  # For evaluation metrics

# Global variables
metrics_json = {}


# Function to calculate perplexity
def calculate_perplexity(prompt, response):
    input_text = prompt + " " + response
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Get model output (logits)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # Cross-entropy loss

    # Calculate perplexity
    perplexity = math.exp(loss.item())
    return perplexity


# Function to compute all metrics
def compute_metrics(predictions):
    global metrics_json

    # BLEU Metric
    bleu = load("bleu")
    bleu_score = bleu.compute(predictions=predictions, references=references)
    print("BLEU score:", bleu_score["bleu"])

    # ROUGE Metric
    rouge = load("rouge")
    rouge_score = rouge.compute(predictions=predictions, references=[ref[0] for ref in references])
    print("ROUGE score:", rouge_score)

    # METEOR Metric
    meteor = load("meteor")
    meteor_score = meteor.compute(predictions=predictions, references=[ref[0] for ref in references])
    print("METEOR score:", meteor_score["meteor"])

    # BERTScore
    try:
        bertscore = load("bertscore")
        bertscore_result = bertscore.compute(predictions=predictions, references=[ref[0] for ref in references],
                                             lang="en")
        BERTScore_precision = sum(bertscore_result["precision"]) / len(bertscore_result["precision"])
        BERTScore_recall = sum(bertscore_result["recall"]) / len(bertscore_result["recall"])
        BERTScore_f1 = sum(bertscore_result["f1"]) / len(bertscore_result["f1"])
        print("BERTScore (Precision):", BERTScore_precision)
        print("BERTScore (Recall):", BERTScore_recall)
        print("BERTScore (F1):", BERTScore_f1)
    except ImportError:
        print("BERTScore requires additional installation. Install it via `pip install bert-score`.")
        BERTScore_precision, BERTScore_recall, BERTScore_f1 = None, None, None

    # Collect all metrics into a JSON object
    metrics_json = {
        "bleu": bleu_score["bleu"],
        "rouge1": rouge_score["rouge1"],
        "rouge2": rouge_score["rouge2"],
        "rougeL": rouge_score["rougeL"],
        "rougeLsum": rouge_score["rougeLsum"],
        "meteor_score": meteor_score["meteor"],
        "BERTScore_precision": BERTScore_precision,
        "BERTScore_recall": BERTScore_recall,
        "BERTScore_f1": BERTScore_f1,
    }

    # Save to JSON file
    with open("data/metrics.json", "w") as file:
        json.dump(metrics_json, file, indent=4)


# Display predictions and references
def display_predictions_and_references():
    print("\nPredictions and References:")
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        print(f"Prompt {i + 1}:")
        print(f"Prediction: {pred}")
        print(f"Reference: {ref[0]}\n")


# Extract predictions and references
def extract_predictions_and_metrics():
    predictions = [response[0]["generated_text"] for response in responses]
    # references = [["Artificial intelligence is..."], ["Machine learning is..."]]
    references = [["Addiction alters brain function, affecting reward, motivation, and decision-making pathways."], [
        "Dopamine is a neurotransmitter involved in the brain's reward system, often hijacked by addictive substances."]]  # # this is for custom_dataset_for_addiction_research.json
    return predictions, references



# Evaluate prompts
def evaluate_prompts():
    global responses, prompts
    # prompts = ["What is AI?", "Define machine learning"]
    prompts = ["How does addiction affect the brain?",
               "What role does dopamine play in addiction?"]  # this is for custom_dataset_for_addiction_research.json
    responses = [evaluator(prompt, max_length=50, truncation=True) for prompt in prompts]


# Load model and tokenizer
def load_model_and_tokenizer():
    global evaluator, model, tokenizer
    model_dir = "./results"  # Specify the directory containing your trained model
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    evaluator = pipeline("text-generation", model=model, tokenizer=tokenizer)


# Main script execution
if __name__ == "__main__":
    load_model_and_tokenizer()
    evaluate_prompts()
    predictions, references = extract_predictions_and_metrics()

    # Compute perplexity for each response
    perplexities = [calculate_perplexity(prompt, response[0]["generated_text"])
                    for prompt, response in zip(prompts, responses)]
    print("\nPerplexities:")
    for i, perplexity in enumerate(perplexities):
        print(f"Prompt {i + 1} Perplexity: {perplexity}")

    # Compute and save evaluation metrics
    compute_metrics(predictions)
    display_predictions_and_references()

    print("\nMetrics JSON:")
    print(json.dumps(metrics_json, indent=4))
