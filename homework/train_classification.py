import torch
import torch.optim as optim
import torch.nn as nn
from .datasets.classification_dataset import load_data
from .models import Classifier, save_model
from pathlib import Path

# Set the dataset path for Colab
train_path = "/content/drive/MyDrive/homework3_original/classification_data/train"
val_path = "/content/drive/MyDrive/homework3_original/classification_data/val"

# Hyperparameters
epochs = 45
batch_size = 64
learning_rate = 1e-3

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the data
train_loader = load_data(dataset_path=train_path, batch_size=batch_size)
val_loader = load_data(dataset_path=val_path, batch_size=batch_size)
# Initialize the classifier model
model = Classifier(in_channels=3, num_classes=6).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

HOMEWORK_DIR = Path(__file__).resolve().parent  # Define the HOMEWORK_DIR

def save_model(model: torch.nn.Module) -> str:
    model_name = "classifier"
    output_path = HOMEWORK_DIR / f"{model_name}.th"
    print(f"Saving model to: {output_path}")  # Debug statement to verify save path
    torch.save(model.state_dict(), output_path)  # Save the model

    return output_path

def train_classification():
    for epoch in range(epochs):
        # Training phase
        model.train()  # Set model to training mode
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Training Loss: {total_loss / len(train_loader)}")

        # Validation phase
        model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                preds = logits.argmax(dim=1)  # Get predicted classes
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy:.4f}")

    # Save the trained model
    save_model(model)

if __name__ == "__main__":
    train_classification()