import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from tkinter import Tk, filedialog
from Segmentation import segment_and_save

# ========== CNN MODEL ==========
class CancerClassifierCNN(nn.Module):
    def __init__(self):
        super(CancerClassifierCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# ========== Load Trained Model ==========
model = CancerClassifierCNN()
model.load_state_dict(torch.load("cancer_classifier.pth", map_location=torch.device('cpu')))
model.eval()

# ========== Transform ==========
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ========== Upload Image ==========
print("🧪 Running: Upload Image Prediction")
Tk().withdraw()
input_path = filedialog.askopenfilename(title="Select an image to predict")
if not input_path:
    print("❌ No image selected.")
    exit()

# ========== Segmentation ==========
segmented_path = "segmented.jpg"
segment_and_save(input_path, segmented_path)

if not os.path.exists(segmented_path):
    print("⚠️ No lesion detected. Try a clearer image.")
    exit()

# ========== Prediction ==========
image = Image.open(segmented_path).convert("RGB")
image = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(image)
    prob = output.item()
    prediction = 1 if prob > 0.5 else 0
    label = "Benign" if prediction == 1 else "Malignant"
    print(f"\n✅ Prediction: {label}")
    print(f"🧠 Confidence: {prob * 100:.2f}% (Benign), {(1 - prob) * 100:.2f}% (Malignant)")
