import os
import cv2
import numpy as np

def segment_and_save(image_path, output_path):
    print(f"\n🔍 Processing: {image_path}")
    
    image = cv2.imread(image_path)

    if image is None:
        print("❌ Could not read:", image_path)
        return

    # === 1. Denoise ===
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # === 2. Contrast Enhancement (CLAHE) ===
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    image = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # === 3. Sharpening ===
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)

    print("🖼️ Image shape:", image.shape)

    # === 4. Grayscale & Thresholding ===
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Debug mask
    debug_mask_path = output_path.replace('.jpg', '_mask.jpg')
    cv2.imwrite(debug_mask_path, mask)

    # === 5. Contour Detection ===
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("🔍 Contours found:", len(contours))

    if not contours:
        print("⚠️ No contours found")
        return

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    print("📐 Largest area:", area)

    if area < 100:
        print("⚠️ Area too small, skipping")
        return

    x, y, w, h = cv2.boundingRect(largest)
    lesion_crop = image[y:y + h, x:x + w]
    cv2.imwrite(output_path, lesion_crop)
    print("✅ Saved segmented image:", output_path)

# ===============================
# 🧠 Run only if script is executed directly
# ===============================
if __name__ == "__main__":
    cancer_dir = 'data/original_images/CANCER/'
    non_cancer_dir = 'data/original_images/NON CANCER/'
    output_dir = 'data/segmented_lesions/'
    os.makedirs(output_dir, exist_ok=True)

    # Optional: test write
    cv2.imwrite(os.path.join(output_dir, 'test_black.jpg'), np.zeros((100, 100, 3)))

    print("\n🔁 Processing CANCER images...")
    for file in os.listdir(cancer_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(cancer_dir, file)
            name = os.path.splitext(file)[0]
            output_path = os.path.join(output_dir, f"M_{name}.jpg")
            segment_and_save(input_path, output_path)

    print("\n🔁 Processing NON-CANCER images...")
    for file in os.listdir(non_cancer_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(non_cancer_dir, file)
            name = os.path.splitext(file)[0]
            output_path = os.path.join(output_dir, f"B_{name}.jpg")
            segment_and_save(input_path, output_path)
