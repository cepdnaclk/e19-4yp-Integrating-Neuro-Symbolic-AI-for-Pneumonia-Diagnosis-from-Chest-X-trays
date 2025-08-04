import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, GlobalAveragePooling2D,
                                   Attention, Concatenate, Dropout,
                                   BatchNormalization, GaussianNoise)
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.metrics import AUC, Recall, Precision
import cv2
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                           roc_curve, auc, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import random

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# Configuration
CSV_DIR = "/storage/projects3/e19-4yp-neuro-symbolic-xray/pneumonia_research"
IMG_DIR = "/storage/projects3/e19-4yp-neuro-symbolic-xray/pneumonia_classification/organized_images"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

# Create unique run identifier
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join(CSV_DIR, f"run_{RUN_ID}")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load data
train_df = pd.read_csv(f"{CSV_DIR}/features_train.csv")
test_df = pd.read_csv(f"{CSV_DIR}/features_test.csv")
val_df = pd.read_csv(f"{CSV_DIR}/features_val.csv")

def load_and_preprocess_image(image_path, augment=True):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        img = cv2.resize(img, IMG_SIZE)
        img = np.expand_dims(img, axis=-1)
        img = np.repeat(img, 3, axis=-1)
        img = img / 255.0

        if augment:
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 0)
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-15, 15)
                M = cv2.getRotationMatrix2D((IMG_SIZE[0]//2, IMG_SIZE[1]//2), angle, 1.0)
                img = cv2.warpAffine(img, M, IMG_SIZE)

        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def process_data(df, split, augment=False):
    images = []
    features = []
    labels = []

    for _, row in df.iterrows():
        image_id = row['ImageID']
        if not image_id.endswith('.png'):
            image_id += '.png'

        image_path = os.path.join(IMG_DIR, split, row['Binary_Label'], image_id)
        img = load_and_preprocess_image(image_path, augment)

        if img is not None:
            images.append(img)
            features.append(row[['opacity', 'texture', 'edge_density',
                               'lung_variance', 'texture_complexity']].values)
            labels.append(row['label'])

    return np.array(images), np.array(features), np.array(labels)

# Process data
X_train_img, X_train_feat, y_train = process_data(train_df, 'train', augment=True)
X_val_img, X_val_feat, y_val = process_data(val_df, 'val')
X_test_img, X_test_feat, y_test = process_data(test_df, 'test')

# Normalize features
scaler = StandardScaler()
X_train_feat = scaler.fit_transform(X_train_feat)
X_val_feat = scaler.transform(X_val_feat)
X_test_feat = scaler.transform(X_test_feat)

# Model building function
def build_high_accuracy_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False,
                              input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    for layer in base_model.layers[:100]:
        layer.trainable = False
    for layer in base_model.layers[100:]:
        layer.trainable = True

    image_input = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = GaussianNoise(0.1)(image_input)
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)

    feature_input = Input(shape=(5,))
    f = Dense(64, activation='swish', kernel_regularizer=l1_l2(0.01, 0.01))(feature_input)
    f = BatchNormalization()(f)
    f = Dropout(0.4)(f)
    f = Dense(32, activation='swish')(f)

        attention1 = Attention()([x, x])
    attention2 = Attention()([f, f])
    x = Concatenate()([x, attention1, f, attention2])

    x = Dense(256, activation='swish', kernel_regularizer=l1_l2(0.01, 0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='swish')(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[image_input, feature_input], outputs=output)

    optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc'), Recall(name='recall'), Precision(name='precision')]
    )

    return model
  GNU nano 6.2                                        enhanced_model.py                                                 # Callbacks
callbacks = [
    EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

# Build and train model
model = build_high_accuracy_model()

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: weight for i, weight in enumerate(class_weights)}

history = model.fit(
    [X_train_img, X_train_feat],
    y_train,
    validation_data=([X_val_img, X_val_feat], y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

  GNU nano 6.2                                        enhanced_model.py                                                 # Generate evaluation report with all visualizations
def generate_full_evaluation(model, test_images, test_features, test_labels, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate predictions
    y_pred = model.predict([test_images, test_features])
    y_pred_classes = (y_pred > 0.5).astype(int)

    # 1. Classification Report
    report = classification_report(test_labels, y_pred_classes, target_names=['Normal', 'Pneumonia'], output_dict=True)
    with open(f"{output_dir}/classification_report.json", 'w') as f:
        json.dump(report, f, indent=4)

    print("\nClassification Report:")
    print(classification_report(test_labels, y_pred_classes, target_names=['Normal', 'Pneumonia']))

    # 2. Confusion Matrix
    plt.figure(figsize=(8,6))
    cm = confusion_matrix(test_labels, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Normal', 'Pneumonia'],
               yticklabels=['Normal', 'Pneumonia'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
     plt.xlabel('Predicted Label')
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()
    print(f"\nConfusion matrix saved to {output_dir}/confusion_matrix.png")

    # 3. ROC Curve
    fpr, tpr, thresholds = roc_curve(test_labels, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f"{output_dir}/roc_curve.png")
     plt.close()
    print(f"ROC curve saved to {output_dir}/roc_curve.png (AUC = {roc_auc:.4f})")

    # 4. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(test_labels, y_pred)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.savefig(f"{output_dir}/precision_recall_curve.png")
    plt.close()
    print(f"Precision-Recall curve saved to {output_dir}/precision_recall_curve.png")

    # 5. Prediction Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred[y_test == 0], bins=50, alpha=0.5, label='Normal', color='blue')
    plt.hist(y_pred[y_test == 1], bins=50, alpha=0.5, label='Pneumonia', color='red')
    plt.title('Prediction Distribution')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f"{output_dir}/prediction_distribution.png")
    plt.close()

     # 6. Training History Plots
    if 'history' in globals():
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/training_history.png")
        plt.close()
        print(f"Training history saved to {output_dir}/training_history.png") return {
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_true': test_labels
    }

# Generate all evaluation artifacts
eval_results = generate_full_evaluation(
    model=model,
    test_images=X_test_img,
    test_features=X_test_feat,
    test_labels=y_test,
    output_dir=CSV_DIR
)
# Save final metrics
final_metrics = {
    'accuracy': model.evaluate([X_test_img, X_test_feat], y_test, verbose=0)[1],
    'auc': eval_results['roc_auc'],
    'recall': eval_results['classification_report']['Pneumonia']['recall'],
    'precision': eval_results['classification_report']['Pneumonia']['precision'],
    'f1_score': eval_results['classification_report']['Pneumonia']['f1-score'],
    'confusion_matrix': eval_results['confusion_matrix']
}

with open(f"{CSV_DIR}/final_metrics.json", 'w') as f:
    json.dump(final_metrics, f, indent=4)

print("\n" + "="*80)
print("Final Model Performance Metrics:")
print("="*80)
print(f"Accuracy: {final_metrics['accuracy']:.4f}")
print(f"AUC: {final_metrics['auc']:.4f}")
print(f"Recall (Sensitivity): {final_metrics['recall']:.4f}")
print(f"Precision: {final_metrics['precision']:.4f}")
print(f"F1 Score: {final_metrics['f1_score']:.4f}")
print("\nAll evaluation artifacts saved to:", CSV_DIR)

# Save model without graphviz dependency
model.save(f"{CSV_DIR}/pneumonia_detection_model.h5")

# Text-based model summary
def save_model_summary(model, file_path):
    with open(file_path, 'w') as f:
        def print_to_file(text):
            f.write(text + '\n')

        model.summary(print_fn=print_to_file)
        f.write('\n\nDetailed Layer Information:\n')
        f.write('='*50 + '\n')
        for i, layer in enumerate(model.layers):
            f.write(f"Layer {i+1}: {layer.name}\n")
            f.write(f"  Type: {layer.__class__.__name__}\n")
            f.write(f"  Input shape: {layer.input_shape}\n")
            f.write(f"  Output shape: {layer.output_shape}\n")
            if hasattr(layer, 'activation'):
                f.write(f"  Activation: {layer.activation.__name__}\n")
            if hasattr(layer, 'units'):
                f.write(f"  Units: {layer.units}\n")
            f.write('-'*50 + '\n')

save_model_summary(model, f"{CSV_DIR}/model_architecture.txt")
print("\nModel architecture summary saved to model_architecture.txt")
    