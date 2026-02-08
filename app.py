import streamlit as st
import torch
import json
import os
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Fish Species Classifier",
    page_icon="🐟",
    layout="centered"
)

st.title("🐟 Fish Species Classification System")
st.caption("SimCLR Encoder + Deep Learning Classifier")

# ------------------ Paths ------------------
MODEL_DIR = "models"
ENCODER_PATH = os.path.join(MODEL_DIR, "fish_simclr_encoder.pt")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "fish_final_model.pt")

os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------ Download Models ------------------
@st.cache_resource(show_spinner=True)
def download_and_load_models():
    # Download from HF Dataset repo
    encoder_file = hf_hub_download(
        repo_id="Riad77/fish-species-classifier",
        filename="fish_simclr_encoder.pt",
        repo_type="dataset"
    )

    classifier_file = hf_hub_download(
        repo_id="Riad77/fish-species-classifier",
        filename="fish_final_model.pt",
        repo_type="dataset"
    )

    encoder = torch.load(encoder_file, map_location="cpu")
    classifier = torch.load(classifier_file, map_location="cpu")

    encoder.eval()
    classifier.eval()

    return encoder, classifier

encoder, classifier = download_and_load_models()

# ------------------ Load Class Names ------------------
@st.cache_resource
def load_class_names():
    with open("models/class_names.json", "r") as f:
        return json.load(f)

class_names = load_class_names()

# ------------------ Image Preprocessing ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------ Prediction ------------------
def predict(image):
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = encoder(image)
        logits = classifier(features)
        probs = F.softmax(logits, dim=1)

    top_probs, top_idxs = probs.topk(3, dim=1)

    results = []
    for p, i in zip(top_probs[0], top_idxs[0]):
        results.append({
            "label": class_names[i.item()],
            "confidence": p.item() * 100
        })
    return results

# ------------------ UI ------------------
uploaded_file = st.file_uploader(
    "Upload a fish image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("🔍 Analyzing image..."):
            preds = predict(image)

        st.success("✅ Prediction Successful")

        st.subheader("🎯 Predicted Species")
        st.markdown(
            f"**{preds[0]['label']}**  \nConfidence: **{preds[0]['confidence']:.2f}%**"
        )

        st.subheader("📊 Top-3 Predictions")
        for i, p in enumerate(preds, 1):
            st.write(f"{i}. {p['label']} — {p['confidence']:.2f}%")

    except Exception as e:
        st.error("❌ Error processing image")
        st.exception(e)

# ------------------ Footer ------------------
st.markdown("---")
st.caption("Powered by PyTorch • Hosted on Hugging Face 🤗")
