from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "./results"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

print("Model and tokenizer loaded successfully!")
