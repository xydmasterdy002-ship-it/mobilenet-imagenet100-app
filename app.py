import os
import json
import torch
import torch.nn as nn
from PIL import Image
import streamlit as st
from torchvision import transforms

# =========================================================
# 路徑設定（相對路徑）
# =========================================================
MODEL_PATH = "mobilenet_imagenet100.pth"  # repo 根目錄
CLASS_IDX_PATH = "Labels.json"            # repo 根目錄

# =========================================================
# 讀取 Labels.json
# =========================================================
with open(CLASS_IDX_PATH, "r", encoding="utf-8") as f:
    idx_to_label = json.load(f)
idx_to_label = {int(k): v for k, v in idx_to_label.items()}  # key 轉 int

num_classes = len(idx_to_label)

# =========================================================
# 定義 MobileNet V1 架構（完整自訂）
# =========================================================
class MobileNetV1(nn.Module):
    def __init__(self, num_classes=100, width_mult=1.0, dropout=0.2):
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

        input_channel = int(32 * width_mult)
        last_channel = int(1024 * width_mult)

        self.model = nn.Sequential(
            conv_bn(3, input_channel, 2),
            conv_dw(input_channel, 64, 1),
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

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# =========================================================
# 載入模型權重
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileNetV1(num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# =========================================================
# Streamlit 網頁
# =========================================================
st.title("MobileNet-ImageNet100 預測 Demo (自訓模型)")

uploaded_file = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        top5_prob, top5_idx = probs.topk(5, dim=1)

    st.write("### 預測結果 (Top-5)")
    for i, (prob, idx) in enumerate(zip(top5_prob[0], top5_idx[0])):
        color = "red" if i == 0 else "black"
        st.markdown(f"<span style='color:{color}'>{i+1}. {idx_to_label[idx]} ({prob*100:.2f}%)</span>", unsafe_allow_html=True)