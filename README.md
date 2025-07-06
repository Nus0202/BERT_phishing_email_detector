# AI-Powered Email Threat Detection Using BERT and FAISS

## 📌 Overview

This project presents an AI-driven approach to detect email threats (such as spam and phishing) by leveraging cutting-edge Natural Language Processing (NLP) and similarity search tools. It combines the semantic understanding capabilities of **BERT** with the scalable retrieval power of **FAISS**, and wraps everything in a user-friendly **Streamlit** web interface.

## 🔍 Motivation

Traditional rule-based spam filters often fail to keep up with evolving tactics. This project tackles the challenge using a semantic understanding pipeline to catch subtle and complex threats in email content.

---

## 📐 Architecture

- **Dataset Preprocessing**: Cleans raw email content and formats it for model input.
- **Model Training**: Fine-tunes `bert-large-uncased` on labeled phishing datasets.
- **Threat Detection**: Uses BERT embeddings and optional FAISS similarity for classification.
- **Web Deployment**: Implements an interactive interface via Streamlit for public access.

---

## 📁 Directory Structure

```bash
├── app.py                  # Streamlit frontend
├── model_training.ipynb    # BERT fine-tuning notebook
├── data_preprocessing.py   # Preprocessing script
├── dataset/                # Raw and cleaned data
├── saved_model/            # Fine-tuned BERT model
├── faiss_index/            # FAISS index files (optional)
└── README.md
```

---

## 📊 Datasets

1. **Training**: [Kaggle Phishing Emails](https://www.kaggle.com/datasets/subhajournal/phishingemails)
2. **Testing**: [University of Twente Phishing Validation Dataset](https://research.utwente.nl/en/datasets/phishing-validation-emails-dataset)

---

## 🧠 Model

- **BERT (`bert-large-uncased`)** from HuggingFace
- Trained with:
  - `batch_size = 16`
  - Mixed precision (`fp16`) on Colab GPUs
  - Early stopping to avoid overfitting

### Performance (on SpamAssassin test set):

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 99.83% |
| Precision | 99.78% |
| Recall    | 99.71% |
| FPR       | 0.11%  |

---

## 🚀 Deployment

Launch the app locally:

```bash
streamlit run app.py
```

### Features

- ✅ Real-time email classification
- ✅ Clear spam/ham output with confidence scores
- ✅ Custom “Exit” button to close Streamlit window


---

## ⚠️ Limitations

- Best performance on structured datasets; raw emails may be misclassified.
- FAISS clustering module is included but not active due to compute limitations.
- Requires command-line for startup (can be enhanced with a GUI launcher).

---

## 📈 Future Improvements

- Diversify training data for better generalization.
- Fully integrate FAISS for clustering & anomaly detection.
- Wrap into executable app for non-technical users.

---

## 📚 References

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [FAISS Library](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)
- [Original Inspiration](https://github.com/EstAlvB/Phishing-Detection-with-BERT)

---

## 🙋‍♂️ Author

**Zhengxiao Sun**  
University of Guelph  
[zsun12@uoguelph.ca](mailto:zsun12@uoguelph.ca)
