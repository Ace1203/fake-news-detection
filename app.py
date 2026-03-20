import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "rahulchoulwar/fake-news-detector-projt"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()


labels = [
"pants-fire",
"false",
"barely-true",
"half-true",
"mostly-true",
"true"
]



fake_classes = ["pants-fire", "false", "barely-true"]
real_classes = ["half-true", "mostly-true", "true"]


def predict_news(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()[0]

    pred_idx = np.argmax(probs)
    pred_label = labels[pred_idx]
    confidence = probs[pred_idx]

    if pred_label in fake_classes:
        final_label = "Fake News"
    else:
        final_label = "Real News"

    return pred_label, final_label, confidence, probs




    st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detector")
st.markdown("Detect whether a news statement is **Fake or Real** using BERT model")

# Input box

text = st.text_area("Enter News Statement:", height=150)

# Button

if st.button("Analyze News"):


    if text.strip() == "":
        st.warning("⚠️ Please enter a news statement")
    else:
        pred_label, final_label, confidence, probs = predict_news(text)

    # Main result
    st.subheader("🔍 Result")

    if final_label == "Fake News":
        st.error(f"❌ {final_label}")
    else:
        st.success(f"✅ {final_label}")

    # Detailed prediction
    st.write(f"**Model Prediction:** {pred_label}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # Probability distribution
    st.subheader("📊 Class Probabilities")

    prob_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}
    st.bar_chart(prob_dict)

    # Extra explanation
    st.markdown("### 🧠 Interpretation")
    st.write("""
    - **pants-fire / false / barely-true → Fake**
    - **half-true / mostly-true / true → Real**
    """)
