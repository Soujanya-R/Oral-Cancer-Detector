import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern

# ===== GLCM Features =====
def glcm_features(gray):
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    features = {
        'GLCM_Energy': graycoprops(glcm, 'energy')[0, 0],
        'GLCM_Homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
        'GLCM_Contrast': graycoprops(glcm, 'contrast')[0, 0],
        'GLCM_Correlation': graycoprops(glcm, 'correlation')[0, 0],
    }
    return features

# ===== Color Features =====
def color_stats(image):
    features = {}
    color_spaces = {'RGB': image, 'HSV': cv2.cvtColor(image, cv2.COLOR_BGR2HSV), 'YCbCr': cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)}
    for space, img in color_spaces.items():
        chans = cv2.split(img)
        for i, c in enumerate(['1', '2', '3']):
            features[f'{space}_mean_{c}'] = np.mean(chans[i])
            features[f'{space}_std_{c}'] = np.std(chans[i])
    return features

# ===== Feature Extractor =====
def extract_features(img_path):
    image = cv2.imread(img_path)
    if image is None:
        print("❌ Cannot read:", img_path)
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = {}

    # Add label (M or B)
    filename = os.path.basename(img_path)
    features['label'] = 'malignant' if filename.startswith('M') else 'benign'
    features['filename'] = filename

    # Extract features
    features.update(glcm_features(gray))
    features.update(color_stats(image))

    return features

# ===== Main =====
seg_dir = 'data/segmented_lesions/'
output_csv = 'features.csv'

data = []
for file in os.listdir(seg_dir):
    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(seg_dir, file)
        feat = extract_features(path)
        if feat:
            data.append(feat)

# Save to CSV
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)
print(f"✅ Features saved to {output_csv}")
