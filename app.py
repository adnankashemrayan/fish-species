import streamlit as st
import torch
import json
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Fish Species Classifier",
    page_icon="🐟",
    layout="centered"
)

st.title("🐟 Fish Species Classification System")
st.caption("SimCLR Encoder + Deep Learning Classifier")

# ------------------ Load Class Names ------------------
@st.cache_resource
def load_class_names():
    with open("models/class_names.json", "r") as f:
        return json.load(f)

class_names = load_class_names()

# ------------------ Load Models ------------------
@st.cache_resource
def load_models():
    encoder = torch.load(
        "models/fish_simclr_encoder.pt",
        map_location="cpu"
    )
    classifier = torch.load(
        "models/fish_final_model.pt",
        map_location="cpu"
    )

    encoder.eval()
    classifier.eval()

    return encoder, classifier

encoder, classifier = load_models()

# ------------------ Image Preprocessing ------------------
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------ Prediction Function ------------------
def predict(image):
    image = inference_transform(image).unsqueeze(0)

    with torch.no_grad():
        features = encoder(image)
        logits = classifier(features)
        probs = F.softmax(logits, dim=1)

    top_probs, top_idxs = probs.topk(3, dim=1)

    results = []
    for prob, idx in zip(top_probs[0], top_idxs[0]):
        results.append({
            "label": class_names[idx.item()],
            "confidence": prob.item() * 100
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

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        with st.spinner("🔍 Analyzing image..."):
            predictions = predict(image)

        st.success("✅ Prediction Successful")

        # Main prediction
        st.subheader("🎯 Predicted Species")
        st.markdown(
            f"""
            **{predictions[0]['label']}**  
            Confidence: **{predictions[0]['confidence']:.2f}%**
            """
        )

        # Top-3 predictions
        st.subheader("📊 Top Predictions")
        for i, pred in enumerate(predictions, start=1):
            st.write(
                f"{i}. {pred['label']} — {pred['confidence']:.2f}%"
            )

    except Exception as e:
        st.error("❌ Invalid image or model error.")
        st.exception(e)

# ------------------ Footer ------------------
st.markdown("---")
st.caption(
    "Built with PyTorch & Streamlit | SimCLR-based Fish Classifier"
)

