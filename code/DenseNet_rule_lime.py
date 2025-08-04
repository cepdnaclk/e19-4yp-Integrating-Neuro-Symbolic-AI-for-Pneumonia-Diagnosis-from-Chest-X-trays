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
from lime import lime_image

  GNU nano 6.2                                      DenseNet_rule_lime.py                                               # Set seeds
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

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train_dt, y_train_dt)

# Add DT rule score to all splits (train, val, test)
def apply_dt_rules(row):
    # Simple example based on root node splits from dt
    if row['opacity'] > dt.tree_.threshold[0] and row['texture'] > dt.tree_.threshold[1]:
        return 1
    else:
        return 0

for df_name, df in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
    df['dt_rule_score'] = df.apply(apply_dt_rules, axis=1)
    df.to_csv(f"{CSV_DIR}/features_{df_name}_with_dt_rules.csv", index=False)


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
  GNU nano 6.2                                      DenseNet_rule_lime.py                                               def process_data_with_rules(df, split, augment=False):
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
            dt_scores.append(row['dt_rule_score'])
    return np.array(images), np.array(labels), np.array(dt_scores)

X_train_img, y_train, dt_train_scores = process_data_with_rules(train_df, 'train', augment=True)
X_val_img, y_val, dt_val_scores = process_data_with_rules(val_df, 'val', augment=False)
X_test_img, y_test, dt_test_scores = process_data_with_rules(test_df, 'test', augment=False)

  GNU nano 6.2                                      DenseNet_rule_lime.py                                               # Build DenseNet model (same as before)
def build_densenet_model():
    base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    for layer in base_model.layers[:-20]:
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
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc'), Recall(name='recall'), Precision(name='precision')]
    )
    return model
  GNU nano 6.2                                      DenseNet_rule_lime.py                                               model = build_densenet_model()

# Compute class weights
class_weights_values = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: w for i, w in enumerate(class_weights_values)}

early_stop = EarlyStopping(monitor='val_auc', patience=7, mode='max', restore_best_weights=True)

# Train model
history = model.fit(
    X_train_img, y_train,
    validation_data=(X_val_img, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# Predict CNN probabilities on test
y_pred_prob = model.predict(X_test_img).flatten()

# ======= LIME explanation generation =======
print("Generating LIME explanation...")

explainer = lime_image.LimeImageExplainer()

def predict_fn(images):
    # images are float64, scaled [0,1], shape (N,H,W,3)
    return model.predict(images)

idx_to_explain = 0  # first test image
test_image = X_test_img[idx_to_explain]

explanation = explainer.explain_instance(
    test_image,
    predict_fn,
    top_labels=1,
    hide_color=0,
    num_samples=1000
)

  GNU nano 6.2                                      DenseNet_rule_lime.py                                               temp, mask = explanation.get_image_and_mask(
    label=explanation.top_labels[0],
    positive_only=True,
    num_features=5,
    hide_rest=True
)

plt.figure(figsize=(6,6))
plt.imshow(temp)
plt.title(f"LIME Explanation for Test Image #{idx_to_explain}")
plt.axis('off')
lime_path = os.path.join(RESULTS_DIR, "lime_explanation.png")
plt.savefig(lime_path)
plt.close()

print(f"LIME explanation saved at {lime_path}")

# ======= Ensemble step =======
alpha = 0.7  # weight for CNN
threshold = 0.5

# Combine CNN prob + rule score
final_scores = alpha * y_pred_prob + (1 - alpha) * dt_test_scores
final_preds = (final_scores > threshold).astype(int)

test_df = test_df.iloc[:len(final_preds)].copy()  # ensure alignment
test_df['cnn_prob'] = y_pred_prob
test_df['dt_rule_score'] = dt_test_scores
test_df['ensemble_score'] = final_scores
test_df['ensemble_pred'] = final_preds
test_df.to_csv(os.path.join(RESULTS_DIR, "features_test_ensemble_results.csv"), index=False)

# Evaluate ensemble
ensemble_report = classification_report(y_test, final_preds, target_names=['Normal', 'Pneumonia'], output_dict=True)
with open(os.path.join(RESULTS_DIR, "ensemble_classification_report.json"), "w") as f:
    json.dump(ensemble_report, f, indent=4)

print("Ensemble Classification Report:")
print(classification_report(y_test, final_preds, target_names=['Normal', 'Pneumonia']))

# Confusion matrix plot
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, final_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumon>plt.title('Ensemble Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(RESULTS_DIR, "ensemble_confusion_matrix.png"))
plt.close()

print(f"Results saved to {RESULTS_DIR}")