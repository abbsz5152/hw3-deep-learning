import torch
import torch.optim as optim
import torch.nn as nn
from .datasets.road_dataset import load_data
from .models import Detector, save_model

# Set dataset paths
train_path = "/content/drive/MyDrive/homework3_original/road_data/train"
val_path = "/content/drive/MyDrive/homework3_original/road_data/val"

# Hyperparameters
epochs = 25
batch_size = 16
learning_rate = 1e-3

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the data
train_loader = load_data(dataset_path=train_path, transform_pipeline='aug', batch_size=batch_size)
val_loader = load_data(dataset_path=val_path, transform_pipeline='default', batch_size=batch_size)

# Initialize the detector model
model = Detector(in_channels=3, num_classes=3).to(device)

# Define the loss functions and optimizer
segmentation_loss_fn = nn.CrossEntropyLoss()  # For segmentation logits
depth_loss_fn = nn.L1Loss()  # For depth prediction (L1 Loss)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train_detection():
    for epoch in range(epochs):
        # Training phase
        model.train()  # Set model to training mode
        total_seg_loss, total_depth_loss = 0.0, 0.0

        for batch in train_loader:
            images = batch['image'].to(device)
            segmentation_labels = batch['track'].to(device)  # True segmentation labels
            depth_labels = batch['depth'].to(device)  # True depth labels

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            segmentation_logits, depth_pred = model(images)

            # Compute the losses
            segmentation_loss = segmentation_loss_fn(segmentation_logits, segmentation_labels)
            depth_loss = depth_loss_fn(depth_pred, depth_labels)

            # Combine losses and backpropagate
            total_loss = segmentation_loss + depth_loss
            total_loss.backward()
            optimizer.step()

            # Accumulate losses
            total_seg_loss += segmentation_loss.item()
            total_depth_loss += depth_loss.item()

        # Print epoch loss
        avg_seg_loss = total_seg_loss / len(train_loader)
        avg_depth_loss = total_depth_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Segmentation Loss: {avg_seg_loss:.4f}, Depth Loss: {avg_depth_loss:.4f}")

        # Validation phase
        model.eval()  # Set model to evaluation mode
        correct_pixels, total_pixels, total_depth_error = 0, 0, 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                segmentation_labels = batch['track'].to(device)
                depth_labels = batch['depth'].to(device)

                # Forward pass
                segmentation_logits, depth_pred = model(images)

                # Calculate segmentation accuracy
                _, predicted_segmentation = torch.max(segmentation_logits, dim=1)
                correct_pixels += predicted_segmentation
