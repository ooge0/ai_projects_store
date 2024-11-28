import os

# Disable oneDNN optimizations for TensorFlow warnings (not required for Hugging Face but included for completeness)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import json
import nlpaug.augmenter.word as naw

# Define initial dataset with prompts and responses
original_data = [
    {"prompt": "What is addiction?",
     "response": "Addiction is a chronic brain disorder characterized by compulsive drug use despite harmful consequences."},
    {"prompt": "Define substance use disorder.",
     "response": "Substance use disorder is a condition in which a person has an uncontrolled dependence on a substance."},
    {"prompt": "What are the common symptoms of addiction?",
     "response": "Symptoms include cravings, loss of control, and continuing to use despite adverse effects."},
    {"prompt": "How does addiction affect the brain?",
     "response": "Addiction alters brain function, affecting reward, motivation, and decision-making pathways."},
    {"prompt": "What role does dopamine play in addiction?",
     "response": "Dopamine is a neurotransmitter involved in the brain's reward system, often hijacked by addictive substances."},
]

# Define augmentation methods
paraphraser = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")
synonym_replacer = naw.SynonymAug(aug_src='wordnet')


# Function to augment data
def augment_text(text, aug_type, n=3):
    augmented_texts = set()
    while len(augmented_texts) < n:
        if aug_type == "paraphrase":
            augmented_texts.add(paraphraser.augment(text))
        elif aug_type == "synonym":
            augmented_texts.add(synonym_replacer.augment(text))
    return list(augmented_texts)


# Augment dataset
augmented_data = []
for entry in original_data:
    # Augment prompts and responses
    prompt_variants = augment_text(entry["prompt"], "paraphrase", n=5)
    response_variants = augment_text(entry["response"], "synonym", n=5)

    # Add augmented pairs to the dataset
    for p, r in zip(prompt_variants, response_variants):
        augmented_data.append({"prompt": p, "response": r})

# Save augmented dataset as JSONL
file_path = "data/datasets/custom_dataset_emotion_augmented.json"
with open(file_path, "w", encoding="utf-8") as f:
    for entry in augmented_data:
        json.dump(entry, f)
        f.write("\n")

print(f"Generated {len(augmented_data)} augmented prompt-response pairs and saved to {file_path}.")
