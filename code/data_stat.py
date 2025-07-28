import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# Paths to CSV files
train_csv = "/storage/projects3/e19-4yp-neuro-symbolic-xray/pneumonia_classification/datasets/research_train_balanced.c>test_csv = "/storage/projects3/e19-4yp-neuro-symbolic-xray/pneumonia_classification/datasets/research_test_balanced.csv"val_csv = "/storage/projects3/e19-4yp-neuro-symbolic-xray/pneumonia_classification/datasets/research_validation_balance>
# Load CSV files
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)
val_df = pd.read_csv(val_csv)

# Check column names and data
print("Train CSV Columns:", train_df.columns.tolist())
print("Test CSV Columns:", test_df.columns.tolist())
print("Validation CSV Columns:", val_df.columns.tolist())
print("Train Binary_Label Values:", train_df['Binary_Label'].value_counts().to_dict())
print("Train Updated_Symptoms Sample:", train_df['Updated_Symptoms'].head().to_list())


# Map Binary_Label to numeric labels
train_df['label'] = train_df['Binary_Label'].map({'pneumonia': 1, 'non_pneumonia': 0})
test_df['label'] = test_df['Binary_Label'].map({'pneumonia': 1, 'non_pneumonia': 0})
val_df['label'] = val_df['Binary_Label'].map({'pneumonia': 1, 'non_pneumonia': 0})

# Count images per class
train_counts = train_df['label'].value_counts()
test_counts = test_df['label'].value_counts()
val_counts = val_df['label'].value_counts()

# Create class distribution bar chart
plt.figure(figsize=(10, 6))
datasets = [
    {'label': 'Train', 'data': [train_counts[1] if 1 in train_counts.index else 0, train_counts[0] if 0 in train_counts>
    {'label': 'Validation', 'data': [val_counts[1] if 1 in val_counts.index else 0, val_counts[0] if 0 in val_counts.in>
    {'label': 'Test', 'data': [test_counts[1] if 1 in test_counts.index else 0, test_counts[0] if 0 in test_counts.inde>
]
x = range(len(['Pneumonia', 'Non-Pneumonia']))
width = 0.25
for i, dataset in enumerate(datasets):
    plt.bar([j + i * width for j in x], dataset['data'], width, label=dataset['label'], color=dataset['color'])
plt.xticks([j + width for j in x], ['Pneumonia', 'Non-Pneumonia'])
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("Class Distribution Across Datasets")
plt.legend()
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

# Symptom frequency analysis
train_df['Updated_Symptoms'] = train_df['Updated_Symptoms'].fillna('').str.split(',')
symptoms = train_df['Updated_Symptoms'].explode().value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=symptoms.values, y=symptoms.index)
plt.title("Frequency of Symptoms in Training Set")
plt.xlabel("Count")
plt.ylabel("Symptoms")
plt.tight_layout()
plt.savefig("symptom_frequency.png", dpi=300, bbox_inches='tight')
plt.show()

# Word cloud for symptoms
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(symptoms)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Symptoms")
plt.tight_layout()
plt.savefig("symptom_wordcloud.png", dpi=300, bbox_inches='tight')
plt.show()

# Match images with ImageID (corrected path with suffix matching)
image_dir = "/storage/projects3/e19-4yp-neuro-symbolic-xray/pneumonia_classification/organized_images"
for split_df, split in [(train_df, 'train'), (test_df, 'test'), (val_df, 'val')]:
    image_list = set(os.listdir(os.path.join(image_dir, split, 'pneumonia'))) | set(os.listdir(os.path.join(image_dir, >
    split_df['image_path'] = split_df.apply(lambda row: next((os.path.join(image_dir, split, row['Binary_Label'], f) fo>    split_df['image_exists'] = split_df['image_path'].apply(lambda x: os.path.exists(x) if x else False)
    print(f"{split.capitalize()} Image Existence:", split_df['image_exists'].value_counts())

# Filter for existing images
train_df = train_df[train_df['image_exists']]
test_df = test_df[test_df['image_exists']]
val_df = val_df[val_df['image_exists']]

# Save updated DataFrames
train_df.to_csv("updated_train.csv", index=False)
test_df.to_csv("updated_test.csv", index=False)
val_df.to_csv("updated_val.csv", index=False)

# Count entries
print(f"Number of entries in updated_train.csv: {len(train_df)}")
print(f"Number of entries in updated_test.csv: {len(test_df)}")
print(f"Number of entries in updated_val.csv: {len(val_df)}")