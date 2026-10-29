
import torch
import torch.nn as nn
from torchvision import models


class TSP_ResNet_Policy(nn.Module):

    def __init__(
        self,
        grid_size: int = 5,
        image_size: int = 640,
        pretrained: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        if grid_size < 1:
            raise ValueError(f"grid_size must be >= 1; got {grid_size}")

        self.grid_size = grid_size
        self.image_size = image_size  
        self.freeze_backbone = freeze_backbone

        self.backbone = models.resnet50(pretrained=pretrained)

        self.features = nn.Sequential(*list(self.backbone.children())[:-2])

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.policy_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((grid_size, grid_size)),
            nn.Conv2d(2048, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        logits = self.policy_head(features)
        return logits.flatten(1)


TSP_ResNet50_Policy = TSP_ResNet_Policy


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TSP_ResNet_Policy(grid_size=5, pretrained=False)
    dummy = torch.randn(1, 3, 128, 128)
    output = model(dummy)
    assert output.shape == (1, 25), output.shape
    output.mean().backward()
    assert model.policy_head[1].weight.grad is not None
    print("[OK] ResNet50 SAVD: [1,3,128,128] -> [1,25], backward successful")
