from typing import List
from transformers import BertTokenizer
import re
import pandas as pd

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def load_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = re.sub(r'<.*?>', '', text)       # Remove HTML labels
    text = text.lower()                     # Convert to lower case
    text = re.sub(r'[^a-zA-Z\s]', '', text).strip()  # Remove special characters
    text = re.sub(r'\s+', ' ', text)        # Unify spaces
    return text

def tokenize_text(text: str) -> str:
    tokens = tokenizer.tokenize(text)
    return " ".join(tokens)

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    # Fill missing values in 'Email Text' with empty strings
    data['Email Text'] = data['Email Text'].fillna('')  # Fill NA values
    data['text'] = data['Email Text'].apply(clean_text)  # Clean text
    data['text'] = data['text'].apply(tokenize_text)  # Tokenize text
    return data[['text', 'label']]  # Only keep 'text' and 'label' columns

if __name__ == "__main__":
    # Load Safe Emails and Phishing Emails datasets
    safe_emails_path = '/Users/shawsun/Desktop/Guelph/CIS6020/final_project/dataset/ham/Safe_Emails.csv'
    phishing_emails_path = '/Users/shawsun/Desktop/Guelph/CIS6020/final_project/dataset/spam/Phishing_Emails.csv'
    test_safe_emails_path = '/Users/shawsun/Desktop/Guelph/CIS6020/final_project/dataset/test_ham/Safe_Emails.csv'
    test_phishing_emails_path = '/Users/shawsun/Desktop/Guelph/CIS6020/final_project/dataset/test_spam/Phishing_Emails.csv'

    safe_emails = load_csv(safe_emails_path)
    phishing_emails = load_csv(phishing_emails_path)
    test_safe_emails = load_csv(safe_emails_path)
    test_phishing_emails = load_csv(phishing_emails_path)

    # Add label columns for both datasets
    safe_emails['label'] = 0
    phishing_emails['label'] = 1
    test_safe_emails['label'] = 0
    test_phishing_emails['label'] = 1

    # Combine datasets
    combined_data = pd.concat([safe_emails, phishing_emails], ignore_index=True)
    test_combined_data = pd.concat([test_safe_emails, test_phishing_emails], ignore_index=True)

    # Preprocess data: clean and tokenize
    preprocessed_data = preprocess_data(combined_data)
    test_preprocessed_data = preprocess_data(test_combined_data)

    # Save the preprocessed data to a CSV file
    preprocessed_data.to_csv("preprocessed_emails.csv", index=False)
    preprocessed_data.to_csv("preprocessed_test_emails.csv", index=False)