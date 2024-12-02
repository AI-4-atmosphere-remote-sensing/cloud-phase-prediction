import argparse
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # Correctly import StandardScaler
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from abi_data_utils import load_and_preprocess_data
from abi_model import Deep_coral
import joblib  # For saving the scaler


def train_model(X_train, Y_train, model, optimizer, criterion, epochs=60, save_path="model.pth"):
    """
    Train the model on the training dataset.
    Args:
        X_train (numpy.ndarray): Training data features.
        Y_train (numpy.ndarray): Training data labels.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (torch.nn.Module): Loss function.
        epochs (int): Number of training epochs.
        save_path (str): Path to save the trained model.
    """
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        inputs = torch.tensor(X_train, dtype=torch.float32)
        labels = torch.tensor(Y_train, dtype=torch.long)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def main():
    """
    Main function to load data, train the model, and evaluate performance.
    """
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the .npz file")
    parser.add_argument("--model_path", type=str, default="trained_model.pth", help="Path to save the trained model")
    parser.add_argument("--scaler_path", type=str, default="scaler.pkl", help="Path to save the scaler")
    args = parser.parse_args()

    # Load and preprocess the dataset
    X, Y = load_and_preprocess_data(args.data_path)

    # Fit and save the scaler
    scaler = StandardScaler().fit(X)  # Corrected usage
    joblib.dump(scaler, args.scaler_path)
    print(f"Scaler saved to {args.scaler_path}")

    # Split the dataset into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Define the model, optimizer, and loss function
    model = Deep_coral(input_dim=X_train.shape[1], num_classes=len(np.unique(Y)))
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()

    # Train the model
    train_model(X_train, Y_train, model, optimizer, criterion, save_path=args.model_path)

    # Evaluate the model
    model.eval()
    inputs = torch.tensor(X_test, dtype=torch.float32)
    labels = torch.tensor(Y_test, dtype=torch.long)
    outputs = model(inputs)
    predictions = torch.argmax(outputs, dim=1).numpy()
    accuracy = accuracy_score(Y_test, predictions)
    print(f"Test Accuracy: {accuracy}")


if __name__ == "__main__":
    main()

