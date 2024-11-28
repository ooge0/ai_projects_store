import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get token from environment variable
token = os.getenv("ACCESS_TOKEN")

if token:
    # Add 'Bearer' prefix when sending the token
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)
    print(response.json())
else:
    print("ACCESS_TOKEN is not set.")