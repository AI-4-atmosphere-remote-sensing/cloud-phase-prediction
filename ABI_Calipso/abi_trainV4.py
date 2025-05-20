#ABI_trainV4.py
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, MSELoss
from abi_data_utilsV4 import preProcessing
from abi_modelV4 import Deep_coral

class EarlyStopping:
    """
    Implements early stopping to stop training if validation loss does not improve.
    """
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.bad_epochs = 0

    def step(self, val_loss):
        if self.best_loss is None or val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        return self.bad_epochs >= self.patience

def CORAL(src, tgt):
    """
    Computes the CORAL loss, aligning the covariance matrices of source and target feature distributions.
    """
    d = src.size(1)
    src_c = compute_coral(src)
    tgt_c = compute_coral(tgt)

    loss = torch.sum((src_c - tgt_c) ** 2) / (4 * d * d)
    return loss

def compute_coral(data):
    """
    Computes the covariance matrix for CORAL loss calculation.
    """
    n = data.size(0)
    mean = torch.mean(data, dim=0, keepdim=True)
    data_c = data - mean
    cov = torch.mm(data_c.T, data_c) / (n - 1)
    return cov

def train_model(train_dl, valid_dl, model, n_epochs, lambda_, lambda_l2, device, save_path):
    """
    Trains the DeepCORAL model with classification and domain adaptation losses.
    Includes early stopping and records loss/accuracy curves.
    """

    # Initialize optimizer and loss functions
    optimizer = Adam([
        {'params': model.ddm.parameters(), 'lr': 0.001},
        {'params': model.feature.parameters()},
        {'params': model.fc.parameters(), 'lr': 0.0000005}
    ], lr=0.0000005)

    criterion = CrossEntropyLoss()
    l2loss = MSELoss()

    model.to(device)
    model.train()

    early_stopping = EarlyStopping(patience=5)

    # Lists for tracking training & validation metrics
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_loss_coral = 0
        epoch_loss_classifier_src = 0
        #epoch_loss_classifier_tgt = 0
        epoch_loss_l2 = 0
        correct_train, total_train = 0, 0

        for src_data, src_label, tgt_data, tgt_label in train_dl:
            optimizer.zero_grad()

            src_data, src_label = src_data.to(device), src_label.to(device)
            tgt_data, tgt_label = tgt_data.to(device), tgt_label.to(device)

            # Forward pass
            src_out, tgt_out, dm_out, centr1, centr2 = model(src_data, tgt_data)

            # Compute losses
            loss_classifier_src = criterion(src_out, src_label)
            #loss_classifier_tgt = criterion(tgt_out, tgt_label)
            loss_coral = CORAL(centr1, centr2)
            loss_l2 = l2loss(dm_out, model.ddm(src_data))  # Transform ABI features before computing L2 loss

            # Total loss
            #total_loss = lambda_ * loss_coral + loss_classifier_src + lambda_l2 * loss_l2 + loss_classifier_tgt * 0.5
            total_loss = lambda_ * loss_coral + loss_classifier_src * 2.0 + lambda_l2 * loss_l2
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_loss_coral += loss_coral.item()
            epoch_loss_classifier_src += loss_classifier_src.item()
            #epoch_loss_classifier_tgt += loss_classifier_tgt.item()
            epoch_loss_l2 += loss_l2.item()

            # Compute training accuracy
            correct_train += (torch.argmax(src_out, dim=1) == src_label).sum().item()
            total_train += src_label.size(0)

        # Compute validation loss & accuracy
        val_loss, val_acc = validate_model(valid_dl, model, lambda_, lambda_l2, device)

        # Compute train accuracy
        train_acc = correct_train / total_train

        # Store losses and accuracy
        train_losses.append(epoch_loss / len(train_dl))
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Print results per epoch
        print(f"Epoch {epoch+1}:")
        print(f"  - Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  - Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  - CORAL Loss: {epoch_loss_coral:.4f} | L2 Loss: {epoch_loss_l2:.4f}")

        # Early stopping
        if early_stopping.step(val_loss):
            print("Early stopping triggered. Stopping training.")
            break

    # Save final trained model
    torch.save(model.state_dict(), save_path + "/model.pth")
    print("Model saved successfully!")

    # Save loss & accuracy plots
    plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs, save_path)

def validate_model(valid_dl, model, lambda_, lambda_l2, device):
    """
    Computes validation loss & accuracy.
    """
    model.eval()
    criterion = CrossEntropyLoss()
    l2loss = MSELoss()

    total_loss = 0
    correct, total = 0, 0
    num_batches = 0

    with torch.no_grad():
        for src_data, src_label, tgt_data, tgt_label in valid_dl:
            src_data, src_label = src_data.to(device), src_label.to(device)
            tgt_data, tgt_label = tgt_data.to(device), tgt_label.to(device)

            # Forward pass
            src_out, tgt_out, dm_out, centr1, centr2 = model(src_data, tgt_data)

            # Compute losses
            loss_classifier_src = criterion(src_out, src_label)
            #loss_classifier_tgt = criterion(tgt_out, tgt_label)
            loss_coral = CORAL(centr1, centr2)
            loss_l2 = l2loss(dm_out, model.ddm(src_data))  # Transform ABI features before computing L2 loss

            #total_batch_loss = lambda_ * loss_coral + loss_classifier_src + lambda_l2 * loss_l2 + loss_classifier_tgt * 0.5
            total_batch_loss = lambda_ * loss_coral + loss_classifier_src*2.0 + lambda_l2 * loss_l2
            total_loss += total_batch_loss.item()

            # Compute accuracy
            correct += (torch.argmax(src_out, dim=1) == src_label).sum().item()
            total += src_label.size(0)
            num_batches += 1

    val_loss = total_loss / num_batches
    val_acc = correct / total

    return val_loss, val_acc

def plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs, save_path):
    """
    Plots and saves loss & accuracy curves.
    """
    epochs = range(1, len(train_losses) + 1)

    # Plot Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_path}/loss_curve.png")
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_path}/accuracy_curve.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data_path")
    parser.add_argument("--model_saving_path")
    args = parser.parse_args()

    NUM_LABELS = 3
    EPOCHS = 120
    lambda_ = 0.0001
    lambda_l2 = 0.005
    BATCH_SIZE = 2048


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dl, valid_dl = preProcessing(args.training_data_path, args.model_saving_path, b_size=BATCH_SIZE)

    model = Deep_coral(num_classes=NUM_LABELS)
    train_model(train_dl, valid_dl, model, EPOCHS, lambda_, lambda_l2, device, args.model_saving_path)

