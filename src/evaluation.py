# 1. Accuracy:
# Checks the percentage of total predictions that the model got correct.
# Useful for measuring overall correctness, but can be misleading with imbalanced datasets.

# 2. Precision:
# Out of all customers predicted as churners, how many actually churned?
# High precision = fewer false alarms (False Positives).

# 3. Recall (Sensitivity):
# Out of all customers who actually churned, how many did the model correctly identify?
# High recall = fewer missed churners (False Negatives).

# 4. F1 Score:
# Harmonic mean of Precision and Recall.
# Balances false positives and false negatives.
# Useful when both precision and recall are important, especially with imbalanced data.

# 5. ROC-AUC:
# Measures how well the model distinguishes between churners and non-churners
# across different classification thresholds.
# Higher ROC-AUC = better ability to rank churners above non-churners.

# 6. PR-AUC (Precision-Recall AUC):
# Measures the trade-off between Precision and Recall across different thresholds.
# Particularly useful for imbalanced datasets where identifying the minority class matters.

# 7. Average Precision (AP):
# Summarizes the Precision-Recall curve into a single score using a weighted
# average of precision values at different recall levels.
# Useful for evaluating how effectively the model ranks actual churners.
# This is the metric used by scoring='average_precision' in sklearn.

# 8. Confusion Matrix:
# Shows the number of correct and incorrect predictions for each class.
#
#                 Predicted
#                 0       1
# Actual  0       TN      FP
#         1       FN      TP
#
# TN (True Negative): Correctly predicted non-churners.
# FP (False Positive): Predicted churn, but the customer did not churn.
# FN (False Negative): Actual churner incorrectly predicted as non-churner.
# TP (True Positive): Correctly predicted churners.


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report


def evaluation_engine(y_test, y_pred, y_prob=None):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    if y_prob is not None:
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = roc_auc_score(y_test, y_pred)
    confusion_matrix_ = confusion_matrix(y_test, y_pred)
    classification_report_ = classification_report(y_test, y_pred)

    def displauy_evaluation_results():
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("ROC AUC Score:", roc_auc)
        print("Confusion Matrix:\n", confusion_matrix_)
        print("Classification Report:\n", classification_report_)
    return displauy_evaluation_results()
