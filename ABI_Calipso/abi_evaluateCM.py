import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
from abi_data_utilsCMv2 import preProcessing
from abi_modelCMv2 import Deep_coral

def load_model(model_path, device):
    """
    Load the trained model.
    """
    model = Deep_coral(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_labels(model, dataloader, device):
    """
    Predict labels using the trained model.
    """
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for src_data, src_label, tgt_data, tgt_label in dataloader:
            src_data, src_label = src_data.to(device), src_label.to(device)
            tgt_data, tgt_label = tgt_data.to(device), tgt_label.to(device)

            # Predict ABI and CALIPSO labels
            src_preds, tgt_preds, _, _, _ = model(src_data, tgt_data)

            # Convert to class labels
            tgt_preds = torch.argmax(tgt_preds, dim=1).cpu().numpy()

            # Store predictions and actual labels
            all_preds.append(tgt_preds)  # Only CALIPSO predictions
            all_labels.append(tgt_label.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)

def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    Plot a single confusion matrix with both counts and percentages.
    """
    labels = ["0: Clear", "1: Cloud"]

    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Convert to percentages

    # Create annotation text combining counts and percentages
    annotations = np.array([[f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)" for j in range(cm.shape[1])] for i in range(cm.shape[0])])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=annotations, fmt="", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")

    # Save the plot
    plt.savefig(f"{save_path}/confusion_matrix.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to dataset")
    parser.add_argument("--model_path", required=True, help="Path to trained model (.pth)")
    parser.add_argument("--save_path", required=True, help="Directory to save results")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    _, valid_dl = preProcessing(args.data_path, args.save_path, b_size=2048)

    # Load trained model
    model = load_model(args.model_path, device)

    # Predict labels
    y_pred, y_true = predict_labels(model, valid_dl, device)

    # Compute accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Model Accuracy: {acc:.3f}")

    # Plot and save confusion matrix
    plot_confusion_matrix(y_true, y_pred, args.save_path)

    print(f"Confusion matrix saved at: {args.save_path}/confusion_matrix.png")

