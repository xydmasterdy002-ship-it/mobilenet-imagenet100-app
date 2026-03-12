import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import json
from googletrans import Translator

# ======================================================
# ImageNet Normalize
# ======================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# ======================================================
# MobileNetV1
# ======================================================
class MobileNetV1(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ======================================================
# Load Labels
# ======================================================
labels_path = "Labels.json"
with open(labels_path, "r", encoding="utf-8") as f:
    labels_dict = json.load(f)

# 建立列表索引對應 label 0~99
labels = [v for k, v in labels_dict.items()]

# ======================================================
# Load model
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "mobilenet_imagenet100.pth"

model = MobileNetV1(num_classes=len(labels))
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

# Translator
translator = Translator(service_urls=['translate.google.com'])

# ======================================================
# Streamlit UI
# ======================================================
st.title("MobileNet ImageNet100 Classifier")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Input Image", width=400)

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        prob = torch.softmax(outputs, dim=1)
        top5 = torch.topk(prob, 5)

    st.write("### Top-5 Prediction")
    for i in range(5):
        idx = top5.indices[0][i].item()
        score = top5.values[0][i].item() * 100  # % 表示
        label = labels[idx]
        label_zh = translator.translate(label, dest='zh-tw').text

        if i == 0:
            st.markdown(f"<span style='color:red;'>{i+1}. {label} ({label_zh}) : {score:.2f}%</span>", unsafe_allow_html=True)
        else:
            st.write(f"{i+1}. {label} ({label_zh}) : {score:.2f}%")