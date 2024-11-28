import os
# Disable oneDNN optimizations for TensorFlow warnings (not required for Hugging Face but included for completeness)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# Load environment variables (e.g., Hugging Face token)
load_dotenv()

# Retrieve Hugging Face token from the environment or .env file
token = os.getenv('ACCESS_TOKEN')
if not token:
    raise ValueError("ACCESS_TOKEN is not set. Please set it in your environment or .env file.")

# Load GPT-2 model and tokenizer with token-based authentication
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name, token=token)  # Pretrained GPT-2 model
tokenizer = GPT2Tokenizer.from_pretrained(model_name, token=token)  # Tokenizer for GPT-2
tokenizer.pad_token = tokenizer.unk_token  # Set padding token to unknown token for GPT-2

print("Model and tokenizer loaded successfully!")

# Load the custom dataset from a JSON file
# dataset = load_dataset("json", data_files="data/datasets/custom_dataset.json", split="train")
dataset = load_dataset("json", data_files="data/datasets/custom_dataset_for_addiction_research.json", split="train")

# Define the tokenization function to preprocess data
def tokenize(batch):
    """
    Tokenizes input prompts and prepares them for training.

    - `input_ids`: Tokenized input text for GPT-2.
    - `labels`: GPT-2 requires input text as labels (shifted by one token).
    """
    inputs = tokenizer(batch["prompt"], padding="max_length", truncation=True)
    inputs["labels"] = inputs["input_ids"].copy()  # Use the same tokens as labels for language modeling
    return inputs


# Apply tokenization to the entire dataset
tokenized_data = dataset.map(tokenize, batched=True)

# Define training arguments for the Hugging Face Trainer
training_args = TrainingArguments(
    output_dir="./results",  # Directory to save model and checkpoints
    eval_strategy="no",  # Disable evaluation during training
    learning_rate=5e-5,  # Learning rate for the optimizer
    per_device_train_batch_size=16,  # Batch size per device
    num_train_epochs=3,  # Number of training epochs
    save_total_limit=1,  # Limit the number of saved checkpoints
    logging_dir="./logs"  # Directory for storing logs
)

# Initialize the Hugging Face Trainer
trainer = Trainer(
    model=model,  # The model to train
    args=training_args,  # Training arguments
    train_dataset=tokenized_data  # The tokenized training dataset
)

# Start the training process
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained("./results")  # Save the model in the specified directory
tokenizer.save_pretrained("./results")  # Save the tokenizer in the same directory

print("Training completed and model/tokenizer saved in ./results")
