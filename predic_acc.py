import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score,
    precision_score, f1_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

# Load files
olist_data = pd.read_csv('olist_final_dataset_labeled.csv')
orders = pd.read_csv('olist_orders_dataset_clean.csv')[['order_id', 'customer_id']]
customers = pd.read_csv('olist_customers_dataset_clean.csv')[['customer_id', 'customer_unique_id']]
predicted_data = pd.read_csv('predicted_satisfaction_scores.csv')

# Gắn customer_unique_id vào olist_data, nếu chưa có
olist_data = (
    olist_data
    .merge(orders, on='order_id', how='left')
    .merge(customers, on='customer_id', how='left')
)

if 'customer_unique_id' not in olist_data.columns:
    raise KeyError("customer_unique_id missing after merge. Check orders/customers files.")

# Merge với dự đoán
merged = olist_data.merge(predicted_data, on='customer_unique_id', how='inner')

# true label: 0 cho 1-3, 1 cho 4-5
merged['true_label'] = merged['review_score'].apply(lambda x: 0 if x <= 3 else 1)
merged['pred_label'] = (merged['predicted_satisfaction'] >= 0.5).astype(int)

y_true = merged['true_label']
y_score = merged['predicted_satisfaction']
y_pred = merged['pred_label']

# Metrics
acc = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_score)
recall = recall_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("=== Metrics ===")
print(f"Accuracy : {acc:.4f}")
print(f"AUC      : {auc:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-score : {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Không hài lòng', 'Hài lòng']))
print(f"\nMerged rows: {len(merged)}")

# ROC curve
fpr, tpr, _ = roc_curve(y_true, y_score)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})', color='blue')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.close()

# Precision-Recall curve
precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_score)
plt.figure(figsize=(7, 6))
plt.plot(recall_vals, precision_vals, color='green')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.tight_layout()
plt.savefig('precision_recall_curve.png', dpi=150)
plt.close()

# Confusion matrix heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['pred 0', 'pred 1'],
            yticklabels=['true 0', 'true 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()

print("Đã lưu: roc_curve.png, precision_recall_curve.png, confusion_matrix.png")