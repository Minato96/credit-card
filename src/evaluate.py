from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def print_classification_report(y_true, y_pred):
    """Prints the classification report."""
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Fraud', 'Fraud']))

def plot_and_save_confusion_matrix(model, X_test, y_test, save_path):
    """Plots and saves the confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    matrix = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Fraud', 'Fraud'])
    matrix.plot(ax=ax)
    plt.title(f'Confusion Matrix for {type(model).__name__}')
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")