import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Resolve oneDNN warning
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from evaluate import load  # Use evaluate for BLEU metric

# Load model and tokenizer
model_dir = "./results"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
evaluator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Evaluate prompts
# prompts = ["What is AI?", "Define machine learning"] # this is for custom_dataset.json
prompts = ["How does addiction affect the brain?", "What role does dopamine play in addiction?"] # this is for custom_dataset_for_addiction_research.json
responses = [evaluator(prompt, max_length=50, truncation=True) for prompt in prompts]

# Extract predictions for BLEU metric
predictions = [response[0]["generated_text"] for response in responses]
# references = [["Artificial intelligence is..."], ["Machine learning is..."]]  # this is for custom_dataset.json
references = [["Addiction alters brain function, affecting reward, motivation, and decision-making pathways."], ["Dopamine is a neurotransmitter involved in the brain's reward system, often hijacked by addictive substances."]]  # # this is for custom_dataset_for_addiction_research.json

# Compute BLEU metric
metric = load("bleu")
metric_score = metric.compute(predictions=predictions, references=references)
print("BLEU score:", metric_score["bleu"])
