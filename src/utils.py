import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import os

# 1. Training & Validation Curves
def plot_history(history):
    plt.figure(figsize=(12,4))

    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.title('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.legend()

    plt.show()


# 2. Confusion Matrix
def plot_confusion_matrix(model, dataset, class_names=["Cat 🐱", "Dog 🐶"], save_path="../artifacts/confusion_matrix.png"):
    """Generates confusion matrix from a dataset and saves the plot."""
    y_true = []
    y_pred = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        preds = (preds > 0.5).astype("int32")  # threshold sigmoid
        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Confusion matrix saved to {save_path}")


# 3. Classification Report
def save_classification_report(model, dataset, target_names=["Cat", "Dog"], save_path="../artifacts/classification_report.txt"):
    """Saves precision, recall, f1-score report to file."""
    y_true = []
    y_pred = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        preds = (preds > 0.5).astype("int32")
        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    report = classification_report(y_true, y_pred, target_names=target_names)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(report)

    print(f"✅ Classification report saved to {save_path}")
    print(report)  # also print in console
