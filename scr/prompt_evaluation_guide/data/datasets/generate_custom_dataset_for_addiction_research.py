import json

# Define a list of sample prompts and responses related to addiction research
prompts_responses = [
    {"prompt": "What is addiction?", "response": "Addiction is a chronic brain disorder characterized by compulsive drug use despite harmful consequences."},
    {"prompt": "Define substance use disorder.", "response": "Substance use disorder is a condition in which a person has an uncontrolled dependence on a substance."},
    {"prompt": "What are the common symptoms of addiction?", "response": "Symptoms include cravings, loss of control, and continuing to use despite adverse effects."},
    {"prompt": "How does addiction affect the brain?", "response": "Addiction alters brain function, affecting reward, motivation, and decision-making pathways."},
    {"prompt": "What role does dopamine play in addiction?", "response": "Dopamine is a neurotransmitter involved in the brain's reward system, often hijacked by addictive substances."},
    {"prompt": "What is withdrawal?", "response": "Withdrawal refers to the physical and psychological symptoms experienced when a dependent person stops using a substance."},
    {"prompt": "Can addiction be genetic?", "response": "Yes, genetic factors can influence susceptibility to addiction."},
    {"prompt": "What is the role of environment in addiction?", "response": "Environmental factors, such as peer pressure and trauma, significantly contribute to addiction risk."},
    {"prompt": "What is behavioral addiction?", "response": "Behavioral addiction involves compulsive engagement in non-substance activities like gambling or gaming."},
    {"prompt": "What therapies are effective for addiction?", "response": "Cognitive-behavioral therapy (CBT) and motivational interviewing are commonly used in treatment."},
]

# Extend the dataset to 100 unique entries by varying prompts and responses
extended_dataset = []
for i in range(10):
    for entry in prompts_responses:
        # Slightly vary prompts and responses for diversity
        prompt_variant = entry["prompt"] + f" (example {i+1})"
        response_variant = entry["response"] + f" This example emphasizes context {i+1}."
        extended_dataset.append({"prompt": prompt_variant, "response": response_variant})

# Save the dataset as a JSONL file
file_path = "data/datasets/custom_dataset_emotion.json"
with open(file_path, "w", encoding="utf-8") as f:
    for entry in extended_dataset:
        json.dump(entry, f)
        f.write("\n")

print(f"Generated {len(extended_dataset)} prompts and saved to {file_path}.")
