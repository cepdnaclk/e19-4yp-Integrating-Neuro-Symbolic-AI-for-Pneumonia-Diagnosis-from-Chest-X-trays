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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import random

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

  GNU nano 6.2                                        DenseNet_model.py                                                 # Configuration
CSV_DIR = "/storage/projects3/e19-4yp-neuro-symbolic-xray/pneumonia_research"
IMG_DIR = "/storage/projects3/e19-4yp-neuro-symbolic-xray/pneumonia_classification/organized_images"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

# Create unique run identifier
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join(CSV_DIR, f"run_densenet169_{RUN_ID}")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load data
train_df = pd.read_csv(f"{CSV_DIR}/features_train.csv")
val_df = pd.read_csv(f"{CSV_DIR}/features_val.csv")
test_df = pd.read_csv(f"{CSV_DIR}/features_test.csv")

def load_and_preprocess_image(image_path, augment=True):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        img = cv2.resize(img, IMG_SIZE)
        img = np.expand_dims(img, axis=-1)
        img = np.repeat(img, 3, axis=-1)  # Convert to RGB for DenseNet
        img = img / 255.0
        if augment and np.random.rand() > 0.5:
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def process_data(df, split, augment=False):
    images = []
    labels = []
    for _, row in df.iterrows():
        image_id = row['ImageID']
        if not image_id.endswith('.png'):
            image_id += '.png'
        image_path = os.path.join(IMG_DIR, split, row['Binary_Label'], image_id)
        img = load_and_preprocess_image(image_path, augment)
        if img is not None:
            images.append(img)
            labels.append(row['label'])
    return np.array(images), np.array(labels)

# Process data
X_train_img, y_train = process_data(train_df, 'train', augment=True)
X_val_img, y_val = process_data(val_df, 'val', augment=False)
X_test_img, y_test = process_data(test_df, 'test', augment=False)

# Build DenseNet169 model
def build_densenet_model():
    base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    for layer in base_model.layers[:-20]:  # Fine-tune last 20 layers
        layer.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    image_input = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = GaussianNoise(0.1)(image_input)
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='swish')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=image_input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy', AUC(name='auc'>
    return model
  GNU nano 6.2                                        DenseNet_model.py                                                 # Callbacks
callbacks = [EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True, verbose=1)]

# Build and train model
model = build_densenet_model()
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: weight for i, weight in enumerate(class_weights)}

history = model.fit(X_train_img, y_train, validation_data=(X_val_img, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, cla>
# Evaluate
y_pred = model.predict(X_test_img)
y_pred_classes = (y_pred > 0.5).astype(int)
report = classification_report(y_test, y_pred_classes, target_names=['Normal', 'Pneumonia'], output_dict=True)
with open(f"{RESULTS_DIR}/classification_report.json", 'w') as f:
    json.dump(report, f, indent=4)
print("\nDenseNet169 Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=['Normal', 'Pneumonia']))

  GNU nano 6.2                                        DenseNet_model.py                                                 # Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumon>plt.title('DenseNet169 Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(f"{RESULTS_DIR}/confusion_matrix_denset.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('DenseNet169 ROC Curve')
plt.legend(loc="lower right")
plt.savefig(f"{RESULTS_DIR}/roc_curve_densenet.png")
plt.close()

  GNU nano 6.2                                        DenseNet_model.py                                                 # Prediction Distribution
plt.figure(figsize=(8, 6))
plt.hist(y_pred, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold = 0.5')
plt.title('Prediction Distribution (DenseNet169)')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.legend()
plt.savefig(f"{RESULTS_DIR}/prediction_distribution_densenet.png")
plt.close()

# Final Metrics
final_metrics = {
    'accuracy': model.evaluate(X_test_img, y_test, verbose=0)[1],
    'auc': roc_auc,  # Updated to use ROC AUC
    'recall': report['Pneumonia']['recall'],
    'precision': report['Pneumonia']['precision'],
    'f1_score': report['Pneumonia']['f1-score']
}
with open(f"{RESULTS_DIR}/final_metrics.json", 'w') as f:
    json.dump(final_metrics, f, indent=4)
print("\nDenseNet169 Final Metrics:")
for metric, value in final_metrics.items():
    print(f"{metric.capitalize()}: {value:.4f}")
model.save(f"{RESULTS_DIR}/densenet169_model.h5")