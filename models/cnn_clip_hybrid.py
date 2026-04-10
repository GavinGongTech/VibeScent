import torch
import torch.nn as nn
from torchvision import models
from transformers import CLIPModel

# Concatenation rationale:
# - CNN branch (ResNet-50 layer4) captures local texture, pattern, and fine-grained
#   garment details (fabric weave, print type, stitching) that emerge from spatially
#   precise convolutional filters.
# - CLIP branch (ViT-L/14) captures global semantic style context — it has been
#   trained on image–text pairs and encodes high-level concepts like "formal attire"
#   or "streetwear" that are hard to learn from pixels alone.
# Together they complement each other: local detail + global semantics.

CNN_DIM   = 2048
CLIP_DIM  = 768
FUSED_DIM = CNN_DIM + CLIP_DIM   # 2816
MLP_DIM   = 512

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"


class CNNCLIPHybrid(nn.Module):
    """ResNet-50 + frozen CLIP ViT-L/14 hybrid multi-task outfit classifier."""

    def __init__(self):
        super().__init__()

        # ── CNN branch ────────────────────────────────────────────────────────
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])  # (B, 2048, 1, 1)

        for name, param in resnet.named_parameters():
            if not name.startswith("layer4"):
                param.requires_grad = False

        # ── CLIP branch ───────────────────────────────────────────────────────
        clip = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        self.clip_vision  = clip.vision_model         # pooler_output: (B, 1024)
        self.clip_proj    = clip.visual_projection    # 1024 → 768

        for param in self.clip_vision.parameters():
            param.requires_grad = False
        for param in self.clip_proj.parameters():
            param.requires_grad = False

        # ── Fusion MLP ────────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(FUSED_DIM, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, MLP_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # ── Task heads ────────────────────────────────────────────────────────
        self.formal_head    = nn.Linear(MLP_DIM, 1)
        self.season_head    = nn.Linear(MLP_DIM, 4)
        self.gender_head    = nn.Linear(MLP_DIM, 3)
        self.time_head      = nn.Linear(MLP_DIM, 2)
        self.frequency_head = nn.Linear(MLP_DIM, 2)

        # Loss functions
        self._mse = nn.MSELoss()
        self._ce  = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, 3, 224, 224) — normalised with CLIP mean/std
        Returns:
            dict with keys: formal, season, gender, time, frequency
        """
        # CNN branch: (B, 2048)
        cnn_feats = self.cnn_backbone(x).flatten(1)

        # CLIP branch: (B, 1024) → (B, 768)
        clip_feats = self.clip_proj(self.clip_vision(pixel_values=x).pooler_output)

        # Fusion: concat → (B, 2816) → MLP → (B, 512)
        fused = self.fusion(torch.cat([cnn_feats, clip_feats], dim=1))

        return {
            "formal":    self.formal_head(fused).squeeze(1),   # (B,)
            "season":    self.season_head(fused),               # (B, 4)
            "gender":    self.gender_head(fused),               # (B, 3)
            "time":      self.time_head(fused),                 # (B, 2)
            "frequency": self.frequency_head(fused),            # (B, 2)
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
        formal_loss    = self._mse(output["formal"],   labels["formal"])
        season_loss    = self._ce(output["season"],    labels["season"])
        gender_loss    = self._ce(output["gender"],    labels["gender"])
        time_loss      = self._ce(output["time"],      labels["time"])
        frequency_loss = self._ce(output["frequency"], labels["frequency"])

        total_loss = formal_loss + season_loss + gender_loss + time_loss + frequency_loss

        return {
            "total_loss":     total_loss,
            "formal_loss":    formal_loss,
            "season_loss":    season_loss,
            "gender_loss":    gender_loss,
            "time_loss":      time_loss,
            "frequency_loss": frequency_loss,
        }
