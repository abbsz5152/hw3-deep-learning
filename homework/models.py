from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]

model_path = HOMEWORK_DIR / "classifier.th"
if model_path.exists():
    print("Classifier model file exists.")
else:
    print("Classifier model file is missing. Please ensure it was saved.")

class Classifier(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 6):
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Convolutional layers to extract features
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),  # Output: (B, 16, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: (B, 16, 32, 32)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Output: (B, 32, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: (B, 32, 16, 16)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output: (B, 64, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Output: (B, 64, 8, 8)
        )

        # Fully connected layer to produce final logits
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # Flatten the feature map
            nn.Linear(64 * 8 * 8, 128),  # Output: (B, 128)
            nn.ReLU(),
            nn.Linear(128, num_classes)  # Output: (B, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Pass through convolutional layers
        z = self.conv_layers(z)

        # Pass through fully connected layers
        logits = self.fc_layers(z)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        """
        return self(x).argmax(dim=1)


class Detector(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 3):
        """
        Enhanced model for segmentation and depth estimation.

        Args:
            in_channels: int, number of input channels (e.g., 3 for RGB images)
            num_classes: int, number of output segmentation classes
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Encoder (Downsampling path)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),  # Output: (B, 64, H/2, W/2)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: (B, 128, H/4, W/4)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Output: (B, 256, H/8, W/8)
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Decoder for segmentation (upsampling with skip connections)
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)  # Output: (B, 128, H/4, W/4)
        self.upconv2 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # Output: (B, 64, H/2, W/2)
        self.upconv3 = nn.ConvTranspose2d(128, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)  # Output: (B, num_classes, H, W)

        # Depth estimation head (more layers for better depth estimation)
        self.depth_head1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Output: (B, 128, H/8, W/8)
            nn.ReLU(),
        )
        self.depth_head2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),  # Adding skip connection from encoder2
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),  # Output: (B, 1, H/8, W/8)
            nn.Sigmoid()  # Normalize depth output to (0, 1)
        )

        # Upsampling depth to match input size using bilinear upsampling for smoother results
        self.depth_upsample = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),  # Upsample to original input size
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Normalize the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Downsampling path
        z1 = self.encoder1(z)  # Output: (B, 64, H/2, W/2)
        z2 = self.encoder2(z1)  # Output: (B, 128, H/4, W/4)
        z3 = self.encoder3(z2)  # Output: (B, 256, H/8, W/8)

        # Segmentation path (upsampling with skip connections)
        seg = self.upconv1(z3)  # (B, 128, H/4, W/4)
        seg = torch.cat([seg, z2], dim=1)  # Skip connection from encoder2 (resulting in 128 + 128 = 256 channels)
        seg = self.upconv2(seg)  # (B, 64, H/2, W/2)
        seg = torch.cat([seg, z1], dim=1)  # Skip connection from encoder1 (resulting in 64 + 64 = 128 channels)
        seg = self.upconv3(seg)  # (B, num_classes, H, W)

        # Depth estimation path with skip connections
        depth = self.depth_head1(z3)  # (B, 128, H/8, W/8)
        
        # Align z2 size to match depth size before concatenation
        z2_upsampled = nn.functional.interpolate(z2, size=depth.shape[2:], mode='bilinear', align_corners=True)
        depth = torch.cat([depth, z2_upsampled], dim=1)  # Skip connection from encoder2 (resulting in 128 + 128 = 256 channels)

        depth = self.depth_head2(depth)  # Further refine depth (output: (B, 1, H/8, W/8))
        depth = self.depth_upsample(depth).squeeze(1)  # (B, H, W)

        return seg, depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)  # Get segmentation predictions
        depth = raw_depth

        return pred, depth
        
MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}
def load_model(model_name: str, with_weights: bool = False, **model_kwargs) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
            print(f"Loaded model weights from {model_path}")

        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m

def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path

def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024

def debug_model(batch_size: int = 1):
    """
    Test your model implementation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    debug_model()
