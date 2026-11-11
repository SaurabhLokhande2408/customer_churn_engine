from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix,classification_report

def evaluation_engine(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
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