import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_path = "bert-large-finetuned-phishing"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

# Prediction function
def predict_email(text, tokenizer, model, device):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
    return prediction, probs[0].tolist()

# JavaScript script to close the browser window
JS_CLOSE_WINDOW = """
<script>
    window.close();
</script>
"""

# Streamlit application interface
def main():
    # Load model
    model, tokenizer, device = load_model()

    # Page title
    st.title("Phishing Email Detection")
    st.subheader("Enter an email below to check if it's spam or not.")

    # User input
    email_text = st.text_area("Input Email Content", height=200)

    if st.button("Predict"):
        if email_text.strip():
            # Prediction results
            prediction, probabilities = predict_email(email_text, tokenizer, model, device)
            label = "Spam" if prediction == 1 else "Not Spam"
            st.markdown(f"### Prediction: **{label}**")
            st.write(f"Probabilities: {probabilities}")
        else:
            st.warning("Please enter some email content.")

    # Add exit button
    if st.button("Exit"):
        st.write("Closing the application...")
        # Display JavaScript to close the webpage
        st.markdown(JS_CLOSE_WINDOW, unsafe_allow_html=True)
        # Exit the backend
        os._exit(0)

if __name__ == "__main__":
    main()