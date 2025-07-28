import os
import cv2
import numpy as np
from skimage import measure, filters, feature
import pandas as pd
from pathlib import Path

# Configuration
DATA_DIR = "/storage/projects3/e19-4yp-neuro-symbolic-xray/pneumonia_classification/organized_images"
CSV_DIR = "/storage/projects3/e19-4yp-neuro-symbolic-xray/pneumonia_research"  # Updated to match your directory
IMG_SIZE = (224, 224)

def extract_features(image_path):
    print(f"Attempting to process: {image_path}")
    if not os.path.exists(image_path):
        print(f"Path does not exist: {image_path}")
        return None
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image (cv2 error): {image_path}")
        return None
    print(f"Successfully loaded: {image_path}")
    img = cv2.resize(img, IMG_SIZE)

     # Opacity
    thresh = filters.threshold_otsu(img)
    binary = img > thresh
    opacity = np.mean(img[binary]) if np.any(binary) else 0

    # Texture
    texture = feature.graycomatrix(img, [1], [0], levels=256, symmetric=True, normed=True)
    texture = feature.graycoprops(texture, 'contrast')[0, 0]

    # Edge Density
    edges = feature.canny(img, sigma=2)
    edge_density = np.sum(edges) / (IMG_SIZE[0] * IMG_SIZE[1])

    # Lung Field Variance
    lung_region = img[int(IMG_SIZE[0]*0.2):int(IMG_SIZE[0]*0.8), int(IMG_SIZE[1]*0.2):int(IMG_SIZE[1]*0.8)]
    lung_variance = np.var(lung_region) if lung_region.size > 0 else 0

    # Texture Complexity
    texture_complexity = measure.shannon_entropy(img)
     return {
        'opacity': opacity,
        'texture': texture,
        'edge_density': edge_density,
        'lung_variance': lung_variance,
        'texture_complexity': texture_complexity
    }
for split in ['train', 'test', 'val']:
    df = pd.read_csv(f"{CSV_DIR}/updated_{split}.csv")
    print(f"Processing {split} dataset with {len(df)} rows")
    features = []
    for idx, row in df.iterrows():
        print(f"Checking row {idx}: image_exists = {row['image_exists']}, path = {row['image_path']}")
        if row['image_exists']:
            feat = extract_features(row['image_path'])
            if feat:
                feat.update({'ImageID': row['ImageID'], 'Binary_Label': row['Binary_Label'], 'label': row['label']})
                features.append(feat)
            else:
                print(f"No features extracted for {row['ImageID']} due to processing failure")
        else:
            print(f"Skipping {row['ImageID']} due to image_exists = False")
    features_df = pd.DataFrame(features)
    features_df.to_csv(f"{CSV_DIR}/features_{split}.csv", index=False)
    print(f"Processed {split} features and saved to features_{split}.csv with {len(features_df)} entries")