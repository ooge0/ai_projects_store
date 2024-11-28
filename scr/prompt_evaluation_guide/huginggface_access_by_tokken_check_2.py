import os
from dotenv import load_dotenv
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load environment variables
load_dotenv()

# Retrieve the Hugging Face token from environment variables
token = os.getenv('ACCESS_TOKEN')  # Ensure ACCESS_TOKEN is correctly set in .env or environment
print("Hugging Face token: ", token)
if not token:
    raise ValueError("ACCESS_TOKEN is not set. Please set it in your environment or .env file.")

# Load model and tokenizer with token-based authentication
model_name = "gpt-2"
model = GPT2LMHeadModel.from_pretrained(model_name, use_auth_token=token)
tokenizer = GPT2Tokenizer.from_pretrained(model_name, use_auth_token=token)

print("Model and tokenizer loaded successfully!")
