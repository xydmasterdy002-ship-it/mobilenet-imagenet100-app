import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
from googletrans import Translator

# -------------------------------
# 參數設定
# -------------------------------
MODEL_PATH = "mobilenet_imagenet100.pth"
CLASS_IDX_PATH = "class_to_idx.json"
TOPK = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 讀取 class_to_idx
# -------------------------------
with open(CLASS_IDX_PATH, "r", encoding="utf-8") as f:
    class_to_idx = json.load(f)
idx_to_label = {v: k for k, v in class_to_idx.items()}

# -------------------------------
# 定義 MobileNetV1
# -------------------------------
num_classes = len(idx_to_label)

try:
    model = models.mobilenet_v1(weights=None)
except:
    class MobileNetV1(nn.Module):
        def __init__(self, num_classes=100):
            super().__init__()
            from torchvision.models.mobilenetv2 import _make_divisible
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
            self.dropout = nn.Dropout(p=0.2)
            self.fc = nn.Linear(1024, num_classes)
        def forward(self, x):
            x = self.model(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = self.fc(x)
            return x
    model = MobileNetV1(num_classes=num_classes)

model = nn.DataParallel(model).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -------------------------------
# 定義預處理
# -------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# -------------------------------
# Google Translator
# -------------------------------
translator = Translator()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("MobileNet-ImageNet100 Web App")

uploaded_file = st.file_uploader("請上傳圖片", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="上傳圖片", use_column_width=True)

    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)
        top_probs, top_idxs = probs.topk(TOPK)

    st.write("### 預測結果 Top-5")
    for i, (p, idx) in enumerate(zip(top_probs, top_idxs)):
        label_en = idx_to_label[int(idx)]
        label_zh = translator.translate(label_en, dest="zh-tw").text
        prob_percent = p.item() * 100

        if i == 0:
            st.markdown(f"<span style='color:red; font-size:18px'>{i+1}. {label_en} ({label_zh}) - {prob_percent:.2f}%</span>", unsafe_allow_html=True)
        else:
            st.write(f"{i+1}. {label_en} ({label_zh}) - {prob_percent:.2f}%")