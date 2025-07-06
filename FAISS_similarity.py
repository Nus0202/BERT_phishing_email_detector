import faiss
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

# Get device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a function to convert text into vector representations
def text_to_vector(texts, tokenizer, model, device):
    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # Use the [CLS] output
    return embeddings.cpu().numpy()

# Example training data
train_texts = ["This is a spam email", "This is a normal email", "Get rich quickly!"]
train_labels = [1, 0, 1]  # 1 means spam email, 0 means normal email

# Convert training texts into vectors
train_vectors = text_to_vector(train_texts, tokenizer, model, device)

# Build FAISS index
dimension = train_vectors.shape[1]  # Dimension of the vectors
index = faiss.IndexFlatL2(dimension)  # Use L2 distance as the similarity measure
index.add(train_vectors)   # Add vectors to the index

print(f"FAISS Index has {index.ntotal} vectors.")