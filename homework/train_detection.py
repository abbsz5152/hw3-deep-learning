import torch
import torch.optim as optim
import torch.nn as nn
from .datasets.road_dataset import load_data
from .models import Detector, save_model
import numpy as np

# Set dataset paths
train_path = "/content/road_data/train"
val_path = "/content/road_data/val"

# Hyperparameters
epochs = 35  # Increased number of epochs for more training time
batch_size = 10
learning_rate = 1e-3
gradient_clip_val = 5.0
depth_loss_multiplier = 0.011  # Multiplying depth loss by constant factor

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the data
print("Loading training and validation data...")
train_loader = load_data(dataset_path=train_path, transform_pipeline='aug', batch_size=batch_size, num_workers=2, shuffle=True)
val_loader = load_data(dataset_path=val_path, transform_pipeline='default', batch_size=batch_size, num_workers=2, shuffle=False)

# Initialize the detector model
print("Initializing the model...")
model = Detector(in_channels=3, num_classes=3).to(device)

# Define the loss functions and optimizer
class_weights = torch.tensor([0.2, 0.4, 0.4]).to(device)  # Adjust weights for segmentation
segmentation_loss_fn = nn.CrossEntropyLoss(weight=class_weights)  # Weighted segmentation loss

# Custom depth loss function to put more emphasis on large errors
def custom_depth_loss(pred, target):
    abs_diff = torch.abs(pred - target)
    # Use log to put a very strong penalty on larger errors and add linear penalty
    emphasized_loss = torch.mean(abs_diff + 0.8 * abs_diff ** 2 + torch.log1p(abs_diff))
    return emphasized_loss

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)  # Slower learning rate decay for finer training adjustments

def train_detection():
    print("Training started...")

    # Variable to track the best validation loss
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training phase
        model.train()  # Set model to training mode
        total_seg_loss, total_depth_loss = 0.0, 0.0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            segmentation_labels = batch['track'].to(device)  # True segmentation labels
            depth_labels = batch['depth'].to(device)  # True depth labels

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            segmentation_logits, depth_pred = model(images)

            # Compute the losses
            segmentation_loss = segmentation_loss_fn(segmentation_logits, segmentation_labels)
            depth_loss = custom_depth_loss(depth_pred, depth_labels)

            # Combine losses and backpropagate
            # Multiply depth loss by a constant factor to prioritize depth error
            adjusted_depth_loss = depth_loss * depth_loss_multiplier
            total_loss = segmentation_loss + adjusted_depth_loss
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
            optimizer.step()

            # Accumulate losses
            total_seg_loss += segmentation_loss.item()
            total_depth_loss += adjusted_depth_loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                      f"Segmentation Loss: {segmentation_loss.item():.4f}, Depth Loss: {adjusted_depth_loss.item():.4f}")

        # Print epoch loss
        avg_seg_loss = total_seg_loss / len(train_loader)
        avg_depth_loss = total_depth_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Finished - Avg Segmentation Loss: {avg_seg_loss:.4f}, "
              f"Avg Depth Loss: {avg_depth_loss:.4f}")

        # Validation phase
        model.eval()  # Set model to evaluation mode
        correct_pixels, total_pixels, total_depth_error = 0, 0, 0.0
        val_seg_loss, val_depth_loss, total_iou = 0.0, 0.0, 0.0  # Initialize total_iou

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images = batch['image'].to(device)
                segmentation_labels = batch['track'].to(device)
                depth_labels = batch['depth'].to(device)

                # Forward pass
                segmentation_logits, depth_pred = model(images)

                # Calculate segmentation accuracy
                _, predicted_segmentation = torch.max(segmentation_logits, dim=1)
                correct_pixels += (predicted_segmentation == segmentation_labels).sum().item()
                total_pixels += segmentation_labels.numel()

                # Calculate depth error (Mean Absolute Error)
                depth_error = torch.abs(depth_pred - depth_labels).mean().item()
                total_depth_error += depth_error

                # Stop training depth if the error is below the threshold
                if depth_error < 0.015:
                    print(f"Depth error below threshold at Epoch [{epoch+1}], stopping depth training.")
                    break

                # Calculate validation losses for logging
                val_seg_loss += segmentation_loss_fn(segmentation_logits, segmentation_labels).item()
                val_depth_loss += custom_depth_loss(depth_pred, depth_labels).item()

                # Calculate IoU for the batch
                batch_iou = iou_metric(predicted_segmentation.cpu(), segmentation_labels.cpu(), n_classes=3)
                total_iou += batch_iou

        # Calculate average validation losses and metrics
        avg_val_seg_loss = val_seg_loss / len(val_loader)
        avg_val_depth_loss = val_depth_loss / len(val_loader)
        avg_val_loss = avg_val_seg_loss + avg_val_depth_loss
        avg_depth_mae = total_depth_error / len(val_loader)
        seg_accuracy = 100 * correct_pixels / total_pixels
        avg_iou = total_iou / len(val_loader)

        print(f"Validation - Epoch [{epoch+1}/{epochs}], Segmentation Accuracy: {seg_accuracy:.2f}%, "
              f"Depth MAE: {avg_depth_mae:.4f}, IoU: {avg_iou:.4f}")

        # Save the model if it has the best validation loss so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = save_model(model)
            print(f"Model saved to {model_save_path} with validation loss {best_val_loss:.4f}")

        # Step the scheduler
        scheduler.step()

def iou_metric(predicted, target, n_classes):
    intersection = torch.logical_and(predicted == target, target < n_classes).sum().float()
    union = torch.logical_or(predicted == target, target < n_classes).sum().float()
    if union == 0:
        return 0.0
    else:
        return (intersection / union).item()

if __name__ == "__main__":
    train_detection()
