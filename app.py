import json
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# =========================================================
# 路徑設定
# =========================================================
MODEL_PATH = "mobilenet_imagenet100.pth"
LABEL_PATH = "Labels.json"

# =========================================================
# 裝置
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# MobileNetV1 定義（與訓練相同）
# =========================================================
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

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):

        x = self.model(x)

        x = x.view(x.size(0), -1)

        x = self.dropout(x)

        x = self.fc(x)

        return x


# =========================================================
# 讀取 Labels.json
# =========================================================
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    wnid_to_label = json.load(f)

# 與訓練程式相同排序
class_names = sorted(wnid_to_label.keys())

class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

idx_to_label = {
    idx: wnid_to_label[cls] for cls, idx in class_to_idx.items()
}

num_classes = len(idx_to_label)

# =========================================================
# 建立模型
# =========================================================
model = MobileNetV1(num_classes)

# 讀取權重
state_dict = torch.load(MODEL_PATH, map_location=device)

# 處理 DataParallel
new_state_dict = {
    k.replace("module.", ""): v for k, v in state_dict.items()
}

model.load_state_dict(new_state_dict)

model.to(device)

model.eval()

# =========================================================
# ImageNet preprocessing
# =========================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# =========================================================
# Streamlit UI
# =========================================================
st.title("MobileNet ImageNet-100 Demo")

st.write("Upload an image to classify.")

uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png"]
)

# =========================================================
# 推論
# =========================================================
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Input Image", width=300)

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(x)

        prob = F.softmax(output, dim=1)

        top5_prob, top5_idx = prob.topk(5)

    st.subheader("Top-5 Prediction")

    labels = []
    scores = []

    for i in range(5):

        idx = top5_idx[0][i].item()

        label = idx_to_label[idx]

        score = top5_prob[0][i].item()

        labels.append(label)

        scores.append(score)

        if i == 0:

            st.markdown(
                f"**{i+1}. {label} ({score*100:.2f}%)**"
            )

        else:

            st.write(
                f"{i+1}. {label} ({score*100:.2f}%)"
            )

    # =====================================================
    # 機率條圖
    # =====================================================
    fig, ax = plt.subplots()

    ax.barh(labels[::-1], scores[::-1])

    ax.set_xlabel("Probability")

    st.pyplot(fig)