import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC, Recall, Precision
import cv2
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import random
from sklearn.tree import DecisionTreeClassifier

  GNU nano 6.2                                        DenseNet_rule.py                                                  # Set seeds
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# Config
CSV_DIR = "/storage/projects3/e19-4yp-neuro-symbolic-xray/pneumonia_research"
IMG_DIR = "/storage/projects3/e19-4yp-neuro-symbolic-xray/pneumonia_classification/organized_images"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join(CSV_DIR, f"ensemble_run_{RUN_ID}")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load CSVs
train_df = pd.read_csv(f"{CSV_DIR}/features_train.csv")
val_df = pd.read_csv(f"{CSV_DIR}/features_val.csv")
test_df = pd.read_csv(f"{CSV_DIR}/features_test.csv")

# Decision Tree training on train_df features + label
feature_cols = ['opacity', 'texture', 'edge_density', 'lung_variance', 'texture_complexity']
X_train_dt = train_df[feature_cols]
y_train_dt = train_df['label']

  GNU nano 6.2                                        DenseNet_rule.py                                                  dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train_dt, y_train_dt)

# Add DT rule score to all splits (train, val, test)
def apply_dt_rules(row):
    # Simple example based on root node splits from dt
    # You can expand this with the actual rules
    if row['opacity'] > dt.tree_.threshold[0] and row['texture'] > dt.tree_.threshold[1]:
        return 1
    else:
        return 0

for df_name, df in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
    df['dt_rule_score'] = df.apply(apply_dt_rules, axis=1)
    df.to_csv(f"{CSV_DIR}/features_{df_name}_with_dt_rules.csv", index=False)

# Image loading & preprocessing
def load_and_preprocess_image(image_path, augment=False):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image not found: {image_path}")
        img = cv2.resize(img, IMG_SIZE)
        img = np.expand_dims(img, axis=-1)
        img = np.repeat(img, 3, axis=-1)  # grayscale to RGB
        img = img / 255.0
        if augment and np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
def process_data_with_rules(df, split, augment=False):
    images, labels, dt_scores = [], [], []
    for _, row in df.iterrows():
        image_id = row['ImageID']
        if not image_id.endswith('.png'):
            image_id += '.png'
        image_path = os.path.join(IMG_DIR, split, str(row['Binary_Label']), image_id)
        img = load_and_preprocess_image(image_path, augment)
        if img is not None:
            images.append(img)
            labels.append(row['label'])