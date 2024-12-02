import argparse
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from abi_data_utils import load_and_preprocess_data
from abi_model import Deep_coral
import joblib


def evaluate_model(model, X_test, Y_test):
    """
    Evaluate the model on the test dataset.
    Args:
        model (torch.nn.Module): Trained model.
        X_test (numpy.ndarray): Test features.
        Y_test (numpy.ndarray): Test labels.
    Returns:
        numpy.ndarray: Predicted labels.
        float: Accuracy score.
    """
    model.eval()
    inputs = torch.tensor(X_test, dtype=torch.float32)
    labels = torch.tensor(Y_test, dtype=torch.long)
    outputs = model(inputs)
    predictions = torch.argmax(outputs, dim=1).numpy()
    accuracy = accuracy_score(Y_test, predictions)
    return predictions, accuracy


def plot_confusion_matrix(Y_true, Y_pred, class_labels, output_file):
    """
    Plot and save the confusion matrix with both counts and percentages in each cell.
    Args:
        Y_true (numpy.ndarray): True labels.
        Y_pred (numpy.ndarray): Predicted labels.
        class_labels (list): List of class names.
        output_file (str): Path to save the confusion matrix plot.
    """
    cm = confusion_matrix(Y_true, Y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Combine counts and percentages into a single label per cell
    cm_labels = np.array([
        [f"{count}\n({percent:.1f}%)" if count != 0 else "0"
         for count, percent in zip(row, row_percent)]
        for row, row_percent in zip(cm, cm_percentage)
    ])

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=cm_labels, fmt="", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Confusion matrix saved to {output_file}")
    plt.show()


def main():
    """
    Main function for evaluating the model and plotting the confusion matrix.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, required=True, help="Path to the test .npz file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--scaler_path", type=str, required=True, help="Path to the scaler file")
    parser.add_argument("--output_file", type=str, default="confusion_matrix.png", help="Path to save the confusion matrix plot")
    args = parser.parse_args()

    # Load and preprocess test data
    scaler = joblib.load(args.scaler_path)
    X_test, Y_test = load_and_preprocess_data(args.test_data)
    X_test = scaler.transform(X_test)

    # Define class labels for the confusion matrix
    class_labels = ["Class 0 (clear)", "Class 1 (unknown)", "Class 2 (water)", "Class 3 (ice)"]

    # Load the trained model
    model = Deep_coral(input_dim=X_test.shape[1], num_classes=len(class_labels))
    model.load_state_dict(torch.load(args.model_path))
    print("Model loaded successfully.")

    # Evaluate the model
    Y_pred, accuracy = evaluate_model(model, X_test, Y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Plot and save the confusion matrix
    plot_confusion_matrix(Y_test, Y_pred, class_labels, output_file=args.output_file)


if __name__ == "__main__":
    main()

