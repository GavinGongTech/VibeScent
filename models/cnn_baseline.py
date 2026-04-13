import torch
import torch.nn as nn
from torchvision import models


class CNNBaseline(nn.Module):
    """ResNet-50 multi-task outfit classifier with 5 label heads."""

    def __init__(self, class_weights: dict = None, dropout: float = 0.5):
        super().__init__()

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Drop the original FC layer; keep everything up to the global avg pool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # output: (B, 2048, 1, 1)

        # Dropout before heads — regularises the shared 2048-d representation
        # Pair with weight_decay=1e-4 in the Adam optimizer for best effect
        self.dropout = nn.Dropout(dropout)

        # Embedding projection for retrieval / downstream use
        self.embed_proj = nn.Sequential(nn.Linear(2048, 512), nn.ReLU())

        # 5 task heads
        self.formal_head    = nn.Linear(2048, 3)
        self.season_head    = nn.Linear(2048, 4)
        self.gender_head    = nn.Linear(2048, 3)
        self.time_head      = nn.Linear(2048, 2)
        self.frequency_head = nn.Linear(2048, 2)

        # Freeze everything except layer4 and the heads
        for name, param in backbone.named_parameters():
            if not name.startswith("layer4"):
                param.requires_grad = False

        # Loss functions
        cw = class_weights or {}
        self._ce_formal    = nn.CrossEntropyLoss(weight=cw.get("formal"))
        self._ce_season    = nn.CrossEntropyLoss(weight=cw.get("season"))
        self._ce_gender    = nn.CrossEntropyLoss(weight=cw.get("gender"))
        self._ce_time      = nn.CrossEntropyLoss(weight=cw.get("time"))
        self._ce_frequency = nn.CrossEntropyLoss(weight=cw.get("frequency"))

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return 512-d projected backbone features. (B, 512)"""
        feats = self.backbone(x).flatten(1)
        return self.embed_proj(feats)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, 3, 224, 224)
        Returns:
            dict with keys: formal, season, gender, time, frequency
        """
        feats = self.dropout(self.backbone(x).flatten(1))   # (B, 2048)

        return {
            "formal":    self.formal_head(feats),               # (B, 3)
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
        formal_loss    = self._ce_formal(output["formal"],      labels["formal"])
        season_loss    = self._ce_season(output["season"],       labels["season"])
        gender_loss    = self._ce_gender(output["gender"],       labels["gender"])
        time_loss      = self._ce_time(output["time"],           labels["time"])
        frequency_loss = self._ce_frequency(output["frequency"], labels["frequency"])

        total_loss = formal_loss + season_loss + gender_loss + time_loss + frequency_loss

        return {
            "total_loss":     total_loss,
            "formal_loss":    formal_loss,
            "season_loss":    season_loss,
            "gender_loss":    gender_loss,
            "time_loss":      time_loss,
            "frequency_loss": frequency_loss,
        }
