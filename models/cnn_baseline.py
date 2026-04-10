import torch
import torch.nn as nn
from torchvision import models


class CNNBaseline(nn.Module):
    """ResNet-50 multi-task outfit classifier with 5 label heads."""

    def __init__(self):
        super().__init__()

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Drop the original FC layer; keep everything up to the global avg pool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # output: (B, 2048, 1, 1)

        # 5 task heads
        self.formal_head    = nn.Linear(2048, 1)
        self.season_head    = nn.Linear(2048, 4)
        self.gender_head    = nn.Linear(2048, 3)
        self.time_head      = nn.Linear(2048, 2)
        self.frequency_head = nn.Linear(2048, 2)

        # Freeze everything except layer4 and the heads
        for name, param in backbone.named_parameters():
            if not name.startswith("layer4"):
                param.requires_grad = False

        # Loss functions
        self._mse  = nn.MSELoss()
        self._ce   = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, 3, 224, 224)
        Returns:
            dict with keys: formal, season, gender, time, frequency
        """
        feats = self.backbone(x).flatten(1)   # (B, 2048)

        return {
            "formal":    self.formal_head(feats).squeeze(1),   # (B,)
            "season":    self.season_head(feats),               # (B, 4)
            "gender":    self.gender_head(feats),               # (B, 3)
            "time":      self.time_head(feats),                 # (B, 2)
            "frequency": self.frequency_head(feats),            # (B, 2)
        }

    def get_loss(self, output: dict, labels: dict) -> dict:
        """
        Compute per-task losses and weighted total.

        Args:
            output: dict returned by forward()
            labels: dict of tensors from FashionpediaDataset.__getitem__()
                    formal → float32 scalar, others → long scalar

        Returns:
            dict with total_loss and individual loss values
        """
        formal_loss    = self._mse(output["formal"],    labels["formal"])
        season_loss    = self._ce(output["season"],     labels["season"])
        gender_loss    = self._ce(output["gender"],     labels["gender"])
        time_loss      = self._ce(output["time"],       labels["time"])
        frequency_loss = self._ce(output["frequency"],  labels["frequency"])

        total_loss = formal_loss + season_loss + gender_loss + time_loss + frequency_loss

        return {
            "total_loss":     total_loss,
            "formal_loss":    formal_loss,
            "season_loss":    season_loss,
            "gender_loss":    gender_loss,
            "time_loss":      time_loss,
            "frequency_loss": frequency_loss,
        }
