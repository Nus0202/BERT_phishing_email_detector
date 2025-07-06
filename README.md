# AI-Powered Email Threat Detection Using BERT and FAISS

📌 Overview

This project presents an AI-driven approach to detect email threats (such as spam and phishing) by leveraging cutting-edge Natural Language Processing (NLP) and similarity search tools. It combines the semantic understanding capabilities of BERT with the scalable retrieval power of FAISS, and wraps everything in a user-friendly Streamlit web interface.

📐 Architecture
  •	Dataset Preprocessing: Cleans raw email content and formats it for model input.
	•	Model Training: Fine-tunes bert-large-uncased on labeled phishing datasets.
	•	Threat Detection: Uses BERT embeddings and optional FAISS similarity for classification.
	•	Web Deployment: Implements an interactive interface via Streamlit for public access.

📁 Directory Structure
├── app.py                  # Streamlit frontend
├── model_training.ipynb    # BERT fine-tuning notebook
├── data_preprocessing.py   # Preprocessing script
├── dataset/                # Raw and cleaned data
├── saved_model/            # Fine-tuned BERT model
├── faiss_index/            # FAISS index files (optional)
└── README.md

📊 Datasets
1.	Training: Kaggle Phishing Emails
2.	Testing: University of Twente Phishing Validation Dataset

🧠 Model
	•	BERT (bert-large-uncased) from HuggingFace
	•	Trained with:
	•	batch_size = 16
	•	Mixed precision (fp16) on Colab GPUs
	•	Early stopping to avoid overfitting

Performance:
Metrice   |  Value    |
----------------------|
Accuracy  |  99.83%   |
----------------------|
Precision |  99.78%   |
----------------------|
Recal     |  99.71%   |
----------------------|
FPR       |  0.11%    |
-----------------------

🙋‍♂️ Author
Zhengxiao Sun
University of Guelph




