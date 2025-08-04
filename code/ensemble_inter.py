import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Attention, Concatenate
from tensorflow.keras.models import Model
from lime import lime_image
import cv2
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Configuration
IMG_DIR = "/storage/projects3/e19-4yp-neuro-symbolic-xray/pneumonia_classification/organized_images"
CSV_DIR = "/storage/projects3/e19-4yp-neuro-symbolic-xray/pneumonia_research"
IMG_SIZE = (224, 224)

# Load and preprocess image
def load_and_preprocess_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        img = cv2.resize(img, IMG_SIZE)
        img = np.expand_dims(img, axis=-1)
        img = np.repeat(img, 3, axis=-1) / 255.0
        print(f"Preprocessed image shape: {img.shape}")  # Debug shape
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

# Build EfficientNetB0 with attention
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
attention = Attention()([x, x])
x = GlobalAveragePooling2D()(attention)
nn_output = Dense(1, activation='sigmoid', name='nn_output')(x)
nn_model = Model(inputs=base_model.input, outputs=nn_output)

# Assume model is trained (load weights if available)
# nn_model.load_weights("path_to_trained_weights.h5")

# Load rule-based model data
test_df = pd.read_csv(f"{CSV_DIR}/features_test.csv")
if test_df['Binary_Label'].dtype == 'object':
    label_map = {'pneumonia': 1, 'non_pneumonia': 0}
    test_df['Binary_Label'] = test_df['Binary_Label'].str.lower().map(label_map).fillna(1)
feature_names = ['opacity', 'texture', 'edge_density', 'lung_variance', 'texture_complexity']
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(test_df[feature_names], test_df['Binary_Label'])
rule_score = dt.predict_proba(test_df[feature_names])[:, 1]

# Select a valid image
test_dir = os.path.join(IMG_DIR, "test", "pneumonia")
image_files = [f for f in os.listdir(test_dir) if f.endswith('.png')]
if not image_files:
    raise ValueError(f"No .png files found in {test_dir}")
image_path = os.path.join(test_dir, image_files[0])  # e.g., 93846132086346455965795551929741341698_0ln9cz.png
image_filename = os.path.basename(image_path)

# Get the true label from test_df
label_row = test_df[test_df['ImageID'] == image_filename]
true_label = label_row['Binary_Label'].iloc[0] if not label_row.empty else 1
print(f"Using true label {true_label} for {image_filename}")

# Ensemble prediction
img = load_and_preprocess_image(image_path)
if img is not None:
    nn_pred = nn_model.predict(img.reshape(1, 224, 224, 3))[0][0]
    # Match rule_score with image (simplified; assumes first row aligns with first image)
    ensemble_pred = 0.7 * nn_pred + 0.3 * rule_score[0]
    print(f"Neural prediction: {nn_pred:.4f}, Rule-based score: {rule_score[0]:.4f}, Ensemble prediction: {ensemble_pre>
    # Interpretability with LIME on neural component
    explainer = lime_image.LimeImageExplainer()
    predicted_label = 1 if nn_pred > 0.5 else 0
    print(f"Model predicted label: {predicted_label} (probability: {nn_pred:.4f})")
    # Define lambda with explicit reshape check
    def predict_fn(images):
        reshaped_images = np.array([img.reshape(224, 224, 3) for img in images])
        return nn_model.predict(reshaped_images)
    explanation = explainer.explain_instance(img, predict_fn, top_labels=1, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(label=predicted_label, positive_only=True, num_features=5, hide_rest=Tr>    plt.imshow(temp)
    plt.title(f"Ensemble Model - LIME Explanation (Label: {predicted_label})")
       plt.axis('off')
    plt.savefig(f"{CSV_DIR}/ensemble_lime.png")
    plt.close()
    print(f"LIME explanation saved as ensemble_lime.png for {image_filename}")
else:
    print(f"Failed to process image from {test_dir}")