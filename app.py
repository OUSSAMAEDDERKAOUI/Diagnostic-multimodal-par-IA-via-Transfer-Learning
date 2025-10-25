import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
import numpy as np

# -------------------------------
# 🔧 1. App Config
# -------------------------------
st.set_page_config(page_title="Medical Image Classifier", layout="centered")
st.title("🧠 Medical Image Classification")
st.write("Upload an image to predict its class using a fine-tuned GoogLeNet model.")

# -------------------------------
# ⚙️ 2. Device setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 🧩 3. Load model
# -------------------------------
@st.cache_resource
def load_model():
    model = models.googlenet(pretrained=False, aux_logits=False)
    model.fc = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.20),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.20),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 4)
    )
    
    best_model_path = "./Medical-Image-Analysis-AI/US3_training_evaluation_pytorch/models/best_model.pth"  # change path if needed
    model = torch.load(best_model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model

model = load_model()

# -------------------------------
# 🎯 4. Define preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
test_dataset = datasets.ImageFolder("./Medical-Image-Analysis-AI/data/part_one_data/test", transform=transform)
class_names = test_dataset.classes  
# -------------------------------
# 🖼️ 5. Upload image
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width =True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy().flatten()

    pred_idx = np.argmax(probs)
    pred_class = class_names[pred_idx]
    confidence = probs[pred_idx] * 100

    # Display results
    st.markdown(f"### 🏷️ Predicted Class: **{pred_class}**")
    st.markdown(f"**Confidence:** {confidence:.2f}%")

    # Optional: show probabilities for all classes
    st.bar_chart({class_names[i]: float(probs[i]) for i in range(len(class_names))})
