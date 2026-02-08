import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from huggingface_hub import hf_hub_download

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Fish Species Detection",
    page_icon="🐟",
    layout="centered"
)

# ==================================================
# CONFIG
# ==================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "Baim", "Bata", "Batasio (Tenra)", "Chitol", "Croaker (Poya)",
    "Hilsha", "Kajoli", "Meni", "Pabda", "Poli", "Puti",
    "Rita", "Rui", "Rupchada", "Silver Carp", "Tilapia",
    "Common Carp", "Kaikka", "Koral", "Shrimp", "Unknown"
]

NUM_CLASSES = len(CLASS_NAMES)
FEATURE_DIM = 2048

# ==================================================
# LOAD MODELS FROM HUGGING FACE
# ==================================================
@st.cache_resource(show_spinner=True)
def load_models():
    # Download checkpoints
    encoder_path = hf_hub_download(
        repo_id="Riad77/fish-species-classifier",
        filename="fish_simclr_encoder.pt"
    )

    classifier_path = hf_hub_download(
        repo_id="Riad77/fish-species-classifier",
        filename="fish_final_model.pt"
    )

    # Encoder (ResNet50 backbone)
    base = models.resnet50(weights=None)
    encoder = nn.Sequential(*list(base.children())[:-1]).to(DEVICE)

    encoder_state = torch.load(encoder_path, map_location=DEVICE)

    clean_state = {}
    for k, v in encoder_state.items():
        k = k.replace("encoder.", "").replace("backbone.", "").replace("module.", "")
        clean_state[k] = v

    encoder.load_state_dict(clean_state, strict=False)
    encoder.eval()

    # Classifier
    classifier = nn.Linear(FEATURE_DIM, NUM_CLASSES)
    classifier.load_state_dict(
        torch.load(classifier_path, map_location=DEVICE)
    )
    classifier.to(DEVICE)
    classifier.eval()

    # Warm-up
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
        feat = encoder(dummy).view(1, -1)
        _ = classifier(feat)

    return encoder, classifier


encoder, classifier = load_models()

# ==================================================
# IMAGE TRANSFORM
# ==================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==================================================
# PREDICTION FUNCTION
# ==================================================
def predict_topk(image, k=3):
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = encoder(image).view(1, -1)
        logits = classifier(feat)
        probs = torch.softmax(logits, dim=1)[0]

    topk = torch.topk(probs, k)

    results = []
    for idx, cls_idx in enumerate(topk.indices):
        results.append(
            (CLASS_NAMES[cls_idx], float(topk.values[idx] * 100))
        )

    return results

# ==================================================
# UI
# ==================================================
st.markdown(
    "<h1 style='text-align:center;'>🐟 Fish Species Detection</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;color:gray;'>SimCLR + ResNet50 based Fish Classification</p>",
    unsafe_allow_html=True
)
st.markdown("---")

uploaded_file = st.file_uploader(
    "📤 Upload a fish image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Analyze Image"):
            with st.spinner("Analyzing image..."):
                results = predict_topk(image)

            st.markdown("## 🧠 Prediction Results")

            for label, confidence in results:
                st.markdown(f"**{label}**")
                st.progress(int(confidence))
                st.caption(f"Confidence: {confidence:.2f}%")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;font-size:13px;color:gray;'>"
    "© 2026 · Fish AI Classification Platform<br>"
    "Built with PyTorch · SimCLR · Streamlit<br>"
    "Developed by <b>Riad</b>"
    "</p>",
    unsafe_allow_html=True
)
