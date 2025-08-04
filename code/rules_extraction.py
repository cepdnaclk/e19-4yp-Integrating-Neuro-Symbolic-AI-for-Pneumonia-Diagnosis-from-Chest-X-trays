import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
import os

CSV_DIR = "/storage/projects3/e19-4yp-neuro-symbolic-xray/pneumonia_research"

# Load data
train_df = pd.read_csv(f"{CSV_DIR}/features_train.csv")

# Features and labels
X = train_df[['opacity', 'texture', 'edge_density', 'lung_variance', 'texture_complexity']]
y = train_df['label']

# Train decision tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42)  # Limit depth for readability
dt.fit(X, y)

# Extract rules
rules = export_text(dt, feature_names=list(X.columns))
with open(f"{CSV_DIR}/decision_tree_rules.txt", 'w') as f:
    f.write(rules)

print("Decision tree rules saved to decision_tree_rules.txt")
print(rules)

# Optional: Convert rules to a scoring function
def apply_dt_rules(row):
    if row['opacity'] > dt.tree_.threshold[0] and row['texture'] > dt.tree_.threshold[1]:
        return 1  # Example condition from tree
    return 0  # Default

train_df['dt_rule_score'] = train_df.apply(apply_dt_rules, axis=1)
train_df.to_csv(f"{CSV_DIR}/features_train_with_dt_rules.csv", index=False)

for split in ['test', 'val']:
    df = pd.read_csv(f"{CSV_DIR}/features_{split}.csv")
    df['dt_rule_score'] = df.apply(apply_dt_rules, axis=1)
    df.to_csv(f"{CSV_DIR}/features_{split}_with_dt_rules.csv", index=False)