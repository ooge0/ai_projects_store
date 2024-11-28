import os
import requests

# Set token (ensure it's a valid token)
token = os.getenv("ACCESS_TOKEN")

if token:
    headers = {"Authorization": f"Bearer {token}"}
    # Verify token permissions
    response = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)
    print("Token status:", response.json())

    # Check model availability
    model_name = "openai-community/gpt2"
    model_url = f"https://huggingface.co/{model_name}"
    response = requests.get(model_url, headers=headers)
    if response.status_code == 200:
        print(f"Model '{model_name}' is accessible.")
    else:
        print(f"Error accessing model '{model_name}':", response.status_code, response.text)
else:
    print("ACCESS_TOKEN is not set.")
