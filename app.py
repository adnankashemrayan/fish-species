import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from huggingface_hub import hf_hub_download
import base64

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Fish Species Detection",
    page_icon="🐟",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==================================================
# GLOBAL STYLES (FAIL-SAFE)
# ==================================================
def inject_css(image_path):
    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        bg_css = f"""
        background:
          linear-gradient(rgba(0,0,0,0.72), rgba(0,0,0,0.72)),
          url("data:image/png;base64,{encoded}");
        """
    except FileNotFoundError:
        bg_css = """
        background:
          linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        """

    st.markdown(
        f"""
        <style>
        html, body {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            {bg_css}
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .block-container {{
            max-width: 800px;
            background: rgba(255,255,255,0.12);
            backdrop-filter: blur(24px);
            padding: 3.6rem;
            border-radius: 28px;
            border: 1px solid rgba(255,255,255,0.25);
            box-shadow: 0 35px 100px rgba(0,0,0,0.65);
        }}

        button {{
            width: 100%;
            height: 3.5em;
            border-radius: 18px !important;
            font-size: 18px !important;
            font-weight: 600;
            background: linear-gradient(135deg,#00c6ff,#0072ff);
            color: white !important;
            border: none;
        }}

        button:hover {{
            transform: scale(1.03);
            transition: 0.25s ease;
        }}

        section[data-testid="stFileUploader"] {{
            background: rgba(0,0,0,0.45);
            border-radius: 18px;
            padding: 22px;
            border: 1px dashed rgba(255,255,255,0.45);
        }}

        .stProgress > div > div {{
            background-image: linear-gradient(90deg,#00c6ff,#0072ff);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

inject_css("assets/watermark.png")

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown("## 🐟 Fish AI Platform")
    language = st.selectbox("🌐 Language", ["English", "বাংলা"])
    enable_explain = st.checkbox("🔬 Enable Explainability (Grad-CAM)", False)
    enable_report = st.checkbox("📄 Enable PDF Report", False)

    st.markdown("""
    ---
    **Model**
    - SimCLR (Self-Supervised)
    - ResNet50 Encoder
    - Linear Evaluation

    **Developer**
    **Riad**
    """)

# ==================================================
# TEXT
# ==================================================
TEXT = {
    "English": {
        "title": "Fish Species Detection",
        "subtitle": "Industry-Grade AI Fish Classification Platform",
        "upload": "📤 Upload a fish image",
        "analyze": "🔍 Analyze Image",
        "results": "Prediction Results"
    },
    "বাংলা": {
        "title": "মাছের প্রজাতি শনাক্তকরণ",
        "subtitle": "ইন্ডাস্ট্রি-গ্রেড AI ফিশ ক্লাসিফিকেশন সিস্টেম",
        "upload": "📤 মাছের ছবি আপলোড করুন",
        "analyze": "🔍 ছবি বিশ্লেষণ করুন",
        "results": "পূর্বাভাসের ফলাফল"
    }
}
T = TEXT[language]

# ==================================================
# CONFIG
# ==================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "Baim","Bata","Batasio (Tenra)","Chitol","Croaker (Poya)",
    "Hilsha","Kajoli","Meni","Pabda","Poli","Puti",
    "Rita","Rui","Rupchada","Silver Carp","Tilapia",
    "Common Carp","Kaikka","Koral","Shrimp","Unknown"
]

NUM_CLASSES = len(CLASS_NAMES)
FEATURE_DIM = 2048

# ==================================================
# LOAD MODELS (ROBUST)
# ==================================================
@st.cache_resource(show_spinner=False)
def load_models():
    encoder_path = hf_hub_download(
        repo_id="Riad77/fish-species-classifier",
        filename="fish_simclr_encoder.pt",
        repo_type="dataset"
    )

    classifier_path = hf_hub_download(
        repo_id="Riad77/fish-species-classifier",
        filename="fish_final_model.pt",
        repo_type="dataset"
    )

    # -------- Encoder --------
    base = models.resnet50(weights=None)
    encoder = nn.Sequential(*list(base.children())[:-1]).to(DEVICE)

    enc_ckpt = torch.load(encoder_path, map_location=DEVICE)
    clean = {}
    for k, v in enc_ckpt.items():
        k = k.replace("encoder.", "").replace("backbone.", "").replace("module.", "")
        clean[k] = v

    encoder.load_state_dict(clean, strict=False)
    encoder.eval()

    # -------- Classifier (UNIVERSAL LOAD) --------
    classifier = nn.Linear(FEATURE_DIM, NUM_CLASSES).to(DEVICE)
    cls_ckpt = torch.load(classifier_path, map_location=DEVICE)

    if isinstance(cls_ckpt, dict):
        if "state_dict" in cls_ckpt:
            classifier.load_state_dict(cls_ckpt["state_dict"])
        elif "model_state" in cls_ckpt:
            classifier.load_state_dict(cls_ckpt["model_state"])
        else:
            classifier.load_state_dict(cls_ckpt)
    else:
        classifier = cls_ckpt.to(DEVICE)

    classifier.eval()

    # Warm-up
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
        _ = classifier(encoder(dummy).view(1, -1))

    return encoder, classifier

encoder, classifier = load_models()

# ==================================================
# TRANSFORM
# ==================================================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# ==================================================
# PREDICT
# ==================================================
def predict_topk(img, k=3):
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = encoder(img).view(1,-1)
        probs = torch.softmax(classifier(feat), dim=1)[0]

    topk = torch.topk(probs, k)
    return [(CLASS_NAMES[i], float(topk.values[idx]*100))
            for idx, i in enumerate(topk.indices)]

# ==================================================
# HEADER
# ==================================================
st.markdown(f"""
<div style="text-align:center;">
    <h1 style="font-size:48px;">🐟 {T["title"]}</h1>
    <p style="font-size:18px; color:#dddddd;">{T["subtitle"]}</p>
</div>
<hr style="margin:32px 0;">
""", unsafe_allow_html=True)

# ==================================================
# MAIN APP
# ==================================================
file = st.file_uploader(T["upload"], type=["jpg","jpeg","png"])

if file:
    try:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button(T["analyze"]):
            with st.spinner("Running deep visual analysis..."):
                results = predict_topk(image)

            st.markdown(f"## 🧠 {T['results']}")

            for label, conf in results:
                st.markdown(f"**{label}**")
                st.progress(int(conf))
                st.caption(f"Confidence: {conf:.2f}%")

            if enable_explain:
                st.info("🔬 Grad-CAM enabled (hook ready).")

            if enable_report:
                st.info("📄 PDF report enabled (hook ready).")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# ==================================================
# FOOTER
# ==================================================
st.markdown("""
<hr style="margin-top:50px;">
<p style="text-align:center; color:#cfcfcf; font-size:14px;">
© 2026 · Fish AI Classification Platform<br>
Built with PyTorch · SimCLR · Streamlit<br>
Developed by <b>Riad</b>
</p>
""", unsafe_allow_html=True)
