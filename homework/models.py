from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.ReLU(),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: (B, 128, H/4, W/4)
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Output: (B, 256, H/8, W/8)
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Decoder for segmentation (upsampling with skip connections)
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)  # Output: (B, 128, H/4, W/4)
        self.upconv2 = nn.ConvTranspose2d(128 + 128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # Output: (B, 64, H/2, W/2)
        self.upconv3 = nn.ConvTranspose2d(64 + 64, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)  # Output: (B, num_classes, H, W)

        # Depth estimation head (using more sophisticated upsampling)
        self.depth_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Output: (B, 128, H/8, W/8)
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # Output: (B, 64, H/8, W/8)
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),  # Output: (B, 1, H/8, W/8)
        )

        # Upsampling for depth with skip connections for refinement
        self.depth_upsample1 = nn.ConvTranspose2d(1, 64, kernel_size=4, stride=2, padding=1)  # Output: (B, 64, H/4, W/4)
        self.depth_upsample2 = nn.ConvTranspose2d(64 + 128, 64, kernel_size=4, stride=2, padding=1)  # Output: (B, 64, H/2, W/2)
        self.depth_upsample3 = nn.ConvTranspose2d(64 + 64, 1, kernel_size=4, stride=2, padding=1)  # Output: (B, 1, H, W)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Normalize the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Downsampling path
        z1 = self.encoder1(z)  # (B, 64, H/2, W/2)
        z2 = self.encoder2(z1)  # (B, 128, H/4, W/4)
        z_enc = self.encoder3(z2)  # (B, 256, H/8, W/8)

        # Segmentation path (upsampling with skip connections)
        seg = self.upconv1(z_enc)
        seg = torch.cat([seg, z2], dim=1)
        seg = self.upconv2(seg)
        seg = torch.cat([seg, z1], dim=1)
        seg = self.upconv3(seg)

        # Depth estimation path
        depth = self.depth_head(z_enc)
        depth = self.depth_upsample1(depth)
        depth = torch.cat([depth, z2], dim=1)
        depth = self.depth_upsample2(depth)
        depth = torch.cat([depth, z1], dim=1)
        depth = self.depth_upsample3(depth).squeeze(1)

        return seg, depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)
        depth = raw_depth

        return pred, depth

MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}

def load_model(model_name: str, with_weights: bool = False, **model_kwargs) -> torch.nn.Module:
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

    model_size_mb = calculate_model_size_mb(m)
    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m

def save_model(model: torch.nn.Module) -> str:
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
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024

def debug_model(batch_size: int = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)
    print(f"Input shape: {sample_batch.shape}")
    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    debug_model()
