
# 🩺 Oral Cancer Detection using Deep Learning (CNN & Image Processing)

This project focuses on building an AI-powered system for early detection of Oral Cancer using Deep Learning and Computer Vision techniques. The model classifies oral cavity images into Benign (Non-Cancerous) or Malignant (Cancerous) categories and provides a confidence score.
The system also integrates image preprocessing, lesion segmentation, and a user-friendly interface for real-time prediction.
## 📷 Screenshots

### Example 1:

1. Upload Image Page

![App Screenshot](https://raw.githubusercontent.com/Soujanya-R/Oral-Cancer-Detector/refs/heads/main/ima/Screenshot%202025-09-03%20202239.png)

2. Image Preview 
![App Screenshot](https://raw.githubusercontent.com/Soujanya-R/Oral-Cancer-Detector/refs/heads/main/ima/Screenshot%202025-09-03%20202123.png)

3. Segmented Mask
![App Screenshot](https://raw.githubusercontent.com/Soujanya-R/Oral-Cancer-Detector/refs/heads/main/ima/Screenshot%202025-09-03%20202116.png)

4. Detection Result – Negative
![App Screenshot](https://raw.githubusercontent.com/Soujanya-R/Oral-Cancer-Detector/refs/heads/main/ima/Screenshot%202025-09-03%20202254.png)

### Example 2:
1. Upload Image Page
![App Screenshot](https://raw.githubusercontent.com/Soujanya-R/Oral-Cancer-Detector/refs/heads/main/ima/Screenshot%202025-09-03%20202506.png)

2. Segmented Mask
![App Screenshot](https://raw.githubusercontent.com/Soujanya-R/Oral-Cancer-Detector/refs/heads/main/ima/Screenshot%202025-09-03%20202545.png)

3. Detection Result – Positive
![App Screenshot](https://raw.githubusercontent.com/Soujanya-R/Oral-Cancer-Detector/refs/heads/main/ima/Screenshot%202025-09-03%20202517.png)

4. Training Plot
![App Screenshot](https://raw.githubusercontent.com/Soujanya-R/Oral-Cancer-Detector/refs/heads/main/ima/Screenshot%202025-09-03%20202138.png)

5.Correlation Heatmap of Selected Features
![App Screenshot](https://raw.githubusercontent.com/Soujanya-R/Oral-Cancer-Detector/refs/heads/main/ima/Screenshot%202025-09-03%20202056.png)
## ✨ Features

- End-to-End Deep Learning Pipeline

  - Image acquisition, preprocessing, segmentation, and classification.

- Image Preprocessing

  - Noise removal, contrast enhancement, sharpening.

- Lesion Segmentation

  - Isolates ROI (Region of Interest) using Otsu Thresholding and contours.

- Deep Learning Model

  - CNN-based binary classifier with high accuracy.

- GUI/CLI Interface

   - Upload an image and get predictions with confidence scores.

##  🛠 Tech Stack



- Programming Language: Python (3.10+)

- Deep Learning Framework: PyTorch, Torchvision

- Image Processing: OpenCV

- Data Handling: NumPy, Pandas

- Environment: Jupyter Notebook / VS Code

- Version Control: Git & GitHub

## 📂 Project Structure

```bash
Oral-Cancer-Detection/
│
├── data/
│   ├── original_images/
│   │   ├── CANCER/
│   │   └── NON-CANCER/
│   ├── segmented_lesions/
│
├── models/
│   └── cancer_classifier.pth
│
├── scripts/
│   ├── train_model.py
│   ├── predict_image.py
│   ├── segment_images.py
│
├── requirements.txt
├── README.md
└── report.pdf
```


## 📥 Dataset


The dataset consists of oral cavity images classified into:

  - CANCER (Malignant)

  - NON-CANCER (Benign)

1. Dataset sourced from open-access medical repositories and research datasets.

2. Images were organized in folders for ImageFolder-based loading in PyTorch.
## 🔍 Methodology
1. Input Image Acquisition

  - Collect oral images from datasets or clinical sources.

2. Preprocessing

- Denoising using fastNlMeansDenoisingColored.

- Contrast enhancement using CLAHE.

- Sharpening using custom kernel filter.

- Resizing and normalization.

3. Segmentation

- Convert to grayscale → Otsu Thresholding → Contour detection.

- Extract Region of Interest (ROI).

- Save segmented lesions for training.

4. Feature Extraction

- CNN-based automatic feature learning.

- Additional handcrafted features:

- GLCM (Gray Level Co-occurrence Matrix).

- GLRLM (Gray Level Run Length Matrix).

5. Classification

- Machine Learning Models Tested:
  -  SVM, KNN, Naïve Bayes (on handcrafted features).

- Deep Learning Model:

  - CNN with:

  - 3 Convolutional Layers

  - ReLU Activation

  - Max Pooling

  - Dropout for regularization

  - Fully Connected Layers

- Loss: Binary Cross-Entropy

- Optimizer: Adam

6. Output

- Prediction: Benign or Malignant

- Confidence Score for each class.
## 📊 Performance Analysis
Machine Learning Models
| Classifier  | Accuracy | Precision | Recall | F1-Score |
| ----------- | -------- | --------- | ------ | -------- |
| SVM         | 85%      | 83%       | 84%    | 83.5%    |
| KNN         | 80%      | 78%       | 79%    | 78.5%    |
| Naive Bayes | 75%      | 72%       | 74%    | 73%      |

Deep Learning (CNN)
| Metric    | Training | Validation |
| --------- | -------- | ---------- |
| Accuracy  | 97%      | 92%        |
| Precision | 93%      | 91%        |
| Recall    | 94%      | 92%        |
| F1-Score  | 93%      | 91.5%      |

- Observation: CNN outperformed all ML models with 92% validation accuracy.

## 🚀 How to Run

1. Clone the Repository
``` bash 
git clone https://github.com/yourusername/oral-cancer-detection.git
cd oral-cancer-detection
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Train the Model
```bash
python train_model.py
```

4. Predict an Image
```bash
python predict_image.py
```
## 📌 Future Enhancements

- Deploy as Web Application using Flask/Django.

- Integrate Transfer Learning with pre-trained models like ResNet.

- Add Explainability (Grad-CAM) for medical interpretability.

- Support Mobile App for Rural Screening.


## 📜 License

This project is for academic purposes only. For commercial use, proper dataset licensing and regulatory approvals are required.
