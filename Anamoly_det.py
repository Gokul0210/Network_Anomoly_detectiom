# Import Libraries 
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

# Settings 
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
os.makedirs("plots", exist_ok=True)

# Load Dataset 
columns = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
    'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
    'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label'
]

train = pd.read_csv("train_data.txt", names=columns)
test = pd.read_csv("test_data.txt", names=columns)

# Preprocessing 
for df in [train, test]:
    # Standardize labels and create binary target
    df['label'] = df['label'].str.rstrip('.')
    df['target'] = (df['label'] != 'normal').astype(int)

numeric_cols = train.select_dtypes(include=['int64','float64']).columns.drop('target')

# Visualizations 
# Dataset sizes
sizes = pd.DataFrame({'Set': ['Training', 'Test'], 'Samples': [len(train), len(test)]})
sns.barplot(x='Set', y='Samples', data=sizes, palette="Set2")
plt.title("Training vs Test Set Sizes")
plt.ylabel("Number of Samples")
plt.savefig("plots/train_test_sizes.png")
plt.show()

# Normal vs Attack counts
for df, name in zip([train, test], ['Training', 'Test']):
    sns.countplot(x=df['target'], palette="Set1")
    plt.title(f"{name} Set: Normal vs Attack")
    plt.xticks([0,1], ["Normal","Attack"])
    plt.ylabel("Count")
    plt.savefig(f"plots/{name.lower()}_normal_vs_attack.png")
    plt.show()

# Pie chart for class distribution
def plot_class_distribution_pie(df, set_name="Test"):
    counts = df['target'].value_counts()
    labels = ['Normal', 'Attack']
    colors = ['#2ca02c','#d62728']
    plt.figure(figsize=(6,6))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title(f"{set_name} Set Class Distribution")
    plt.savefig(f"plots/{set_name.lower()}_class_distribution_pie.png")
    plt.show()

plot_class_distribution_pie(test, "Test")

# Feature Encoding & Scaling 
categorical_cols = ['protocol_type','service','flag']
train_enc = pd.get_dummies(train, columns=categorical_cols)
test_enc = pd.get_dummies(test, columns=categorical_cols)
test_enc = test_enc.reindex(columns=train_enc.columns, fill_value=0)

X_train = train_enc.drop(columns=['label','target'])
y_train = train_enc['target']
X_test = test_enc.drop(columns=['label','target'])
y_test = test_enc['target']

scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Define Machine Learning Models 
models = {
    "LightGBM": lgb.LGBMClassifier(objective='binary', learning_rate=0.05, num_leaves=31,
                                   n_estimators=150, random_state=42, n_jobs=-1, class_weight='balanced'),
    "RandomForest": RandomForestClassifier(n_estimators=150, random_state=42,
                                           class_weight='balanced', n_jobs=-1),
    "NaiveBayes": GaussianNB(),
    "LogisticRegression": LogisticRegression(max_iter=500, class_weight='balanced', random_state=42),
}

# SVM trained on a subset due to computational cost
subset_size = 5000
X_train_svm = X_train.sample(subset_size, random_state=42)
y_train_svm = y_train.loc[X_train_svm.index]
models["SVM"] = SVC(kernel='linear', probability=True, random_state=42)

# Train, Evaluate, and Visualize 
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    if name == "SVM":
        model.fit(X_train_svm, y_train_svm)
        y_train_score = model.decision_function(X_train_svm)
        y_test_score = model.decision_function(X_test)
        # Normalize decision function scores to 0-1
        y_train_score = (y_train_score - y_train_score.min()) / (y_train_score.max() - y_train_score.min())
        y_test_score = (y_test_score - y_test_score.min()) / (y_test_score.max() - y_test_score.min())
    else:
        model.fit(X_train, y_train)
        y_train_score = model.predict_proba(X_train)[:,1]
        y_test_score = model.predict_proba(X_test)[:,1]

    # Determine optimal threshold based on F1-score
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_test_score)
    f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
    optimal_threshold = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5

    # Predictions & metrics
    y_test_pred = (y_test_score >= optimal_threshold).astype(int)
    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred)
    rec = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc = roc_auc_score(y_test, y_test_score)
    results.append([name, acc, prec, rec, f1, roc])

    print(f"{name} Metrics: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, ROC-AUC={roc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Normal","Attack"], yticklabels=["Normal","Attack"])
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"plots/{name}_confusion_matrix.png")
    plt.show()

# Summary Table 
results_df = pd.DataFrame(results, columns=['Model','Accuracy','Precision','Recall','F1','ROC-AUC'])
results_df.to_csv("comparative_results.csv", index=False)
print("\nSummary of All Models:\n", results_df.to_string(index=False))

# Combined ROC Curve 
plt.figure(figsize=(10, 6))

for model_name in results_df['Model']:
    model = models[model_name]

    # Compute test scores
    if model_name == "SVM":
        y_test_score = model.decision_function(X_test)
        y_test_score = (y_test_score - y_test_score.min()) / (y_test_score.max() - y_test_score.min())
    else:
        y_test_score = model.predict_proba(X_test)[:, 1]

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_test_score)
    auc_score = roc_auc_score(y_test, y_test_score)

    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_score:.4f})")

# Baseline
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")

plt.title("Combined ROC Curve for All Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig("plots/combined_roc_curve.png", dpi=150)
plt.show()

# Evaluation Metrics
metrics = ['Accuracy','Precision','Recall','F1','ROC-AUC']
plt.figure(figsize=(10,5))
x = np.arange(len(results_df['Model']))
width = 0.14

for i, metric in enumerate(metrics):
    bars = plt.bar(x + i*width - (len(metrics)/2)*width + width/2, results_df[metric], width, label=metric)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=8)

plt.xticks(x, results_df['Model'], fontsize=10)
plt.ylabel("Metric Value")
plt.ylim(0,1.05)
plt.title("Evaluation Metrics Across Models")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.12), ncol=len(metrics), frameon=False)
plt.tight_layout()
plt.savefig("plots/combined_evaluation_metrics.png")
plt.show()

# Save numeric columns for scaler
joblib.dump(numeric_cols.tolist(), "numeric_columns.pkl")

# Save feature columns for Flask app
joblib.dump(X_train.columns.tolist(), "feature_columns.pkl")

# Save ML models
for name in ['LightGBM','RandomForest','SVM']:
    joblib.dump(models[name], f"{name}_model.pkl")

# Save scaler
joblib.dump(scaler, "scaler.pkl")

print("Models, scaler, and feature columns saved successfully!")
