import random
import string
from transformers import pipeline, set_seed
import pandas as pd
import torch
from datasets import Dataset

# Example word lists; expand these as needed
adjectives = ['Great', 'Big', 'Amazing', 'Proud', 'Cool', 'Gentle', 'Fast', 'Curious', 'Bright', 'Shiny']
nouns = ['Lion', 'Runner', 'Jumper', 'Coder', 'Chef', 'Pilot', 'Gamer', 'Reader', 'Writer', 'Driver']
numbers = ['123', '234', '345', '456', '567', '678', '789', '890', '001', '101']

def generate_username():
    adjective = random.choice(adjectives)
    noun = random.choice(nouns)
    number = random.choice(numbers) if random.random() > 0.5 else ''
    return f"{adjective}{noun}{number}"

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Initialize the generator with increased max_length
generator = pipeline('text-generation', model='gpt2', device=device)
set_seed(42)

# More detailed seed texts
seed_texts = [
    "Details of compensation and benefits in the veterinary field have evolved with changes including",
    "Innovations in medical practice management include comprehensive benefits such as",
    "The standard structure of compensation for medical professionals now often features",
    "Advancements in employment benefits in healthcare sectors typically offer",
    "Typical arrangements for veterinary professionals in private practice include",
    "Recent changes in employment benefits in medical fields have led to",
    "Compensation packages for healthcare professionals now frequently encompass",
    "residency programs cost a lot but there are lots of benefits attached to it for the nation",
    "The vet loves and understands animals a lot, they save a lot of them from illnesses",
    "A doctor saves live and they are very important to our society in order to preserve lives"


]

# Function to generate text with unique outputs
def generate_unique_text(seed_text, num_samples=1000, attempts=100):
    unique_texts = set()
    for _ in range(attempts):
        outputs = generator(seed_text, num_return_sequences=min(num_samples, 1000), max_length=2188, truncation=True)  # Adjust max_length if needed
        for out in outputs:
            text = out['generated_text'].strip()
            unique_texts.add(text)
            if len(unique_texts) >= num_samples:
                return list(unique_texts)
    return list(unique_texts)

# Generate comments
generated_comments = []
for seed in seed_texts:
    comments_from_seed = generate_unique_text(seed, num_samples=1000, attempts=100)
    generated_comments.extend(comments_from_seed)

# Create a Dataset object
dataset = Dataset.from_dict({'comments': generated_comments})

# Function to determine label based on the content of the comment
def determine_label(comment):
    comment_lower = comment.lower()
    if 'vet' in comment_lower or 'veterinary' in comment_lower:
        return 'Veterinarian'
    elif 'doctor' in comment_lower or 'medical' in comment_lower:
        if 'resident' in comment_lower or 'residency' in comment_lower:
            return 'Other'
        else:
            return 'Medical Doctor'
    return 'Other'

# Function to generate usernames
def generate_usernames(batch):
    batch['username'] = [generate_username() for _ in range(len(batch['comments']))]
    return batch

# Function to assign labels
def assign_labels(batch):
    batch['labels'] = [determine_label(comment) for comment in batch['comments']]
    return batch

# Apply functions to the dataset
dataset = dataset.map(generate_usernames, batched=True, batch_size=100)
dataset = dataset.map(assign_labels, batched=True, batch_size=100)

# Convert to DataFrame
df_generated = dataset.to_pandas()

# Save to CSV
df_generated.to_csv('generated_detailed_medical_data.csv', index=False)
