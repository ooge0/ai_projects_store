import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # For resolving 'tensorflow' issue: 'I tensorflow/core/util/port.cc:153] oneDNN ...'
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# Load environment variables
load_dotenv()

# Retrieve the Hugging Face token from environment variables
token = os.getenv('ACCESS_TOKEN')  # Ensure ACCESS_TOKEN is correctly set in .env or environment

if not token:
    raise ValueError("ACCESS_TOKEN is not set. Please set it in your environment or .env file.")

# Load model and tokenizer with token-based authentication
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name, token=token)
tokenizer = GPT2Tokenizer.from_pretrained(model_name, token=token)
tokenizer.pad_token = tokenizer.unk_token

print("Model and tokenizer loaded successfully!")

# Load and preprocess dataset
dataset = load_dataset("json", data_files="data/datasets/custom_dataset.json", split="train")

# Define tokenization function
def tokenize(batch):
    # Tokenize the input prompt
    inputs = tokenizer(batch["prompt"], padding="max_length", truncation=True)
    # The labels are the same as input_ids, but shifted by one token
    inputs["labels"] = inputs["input_ids"].copy()  # GPT2 uses the same input as labels
    return inputs

# Apply tokenization to the dataset
tokenized_data = dataset.map(tokenize, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",  # No evaluation
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    save_total_limit=1,
    logging_dir="./logs"
)

# Initialize the Trainer
trainer = Trainer(
    model=model,               # Your model
    args=training_args,        # Training arguments
    train_dataset=tokenized_data  # Training dataset
)

# Start training
trainer.train()










