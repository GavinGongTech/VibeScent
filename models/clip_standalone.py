import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

# CLIP ViT-L/14 produces 768-dimensional image features from its visual encoder.
# These are the [CLS] token embeddings after the final transformer layer,
# projected through CLIP's visual projection head.

MODEL_NAME = "openai/clip-vit-large-patch14"
CLIP_PROJ_DIM = 768     # after visual_projection (vision_model pooler_output is 1024 → visual_proj → 768)
PROJECTION_DIM = 512    # after our projector MLP (768 → 512)


class CLIPStandalone(nn.Module):
    """Frozen CLIP ViT-L/14 image encoder with a projection MLP and 5 task heads."""

    def __init__(self):
        super().__init__()

        clip = CLIPModel.from_pretrained(MODEL_NAME)

        # Keep only the vision model; discard text encoder
        self.vision_model    = clip.vision_model
        self.visual_proj     = clip.visual_projection   # 1024 → 768 in ViT-L/14

        # Freeze entire CLIP image encoder
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.visual_proj.parameters():
            param.requires_grad = False

        # Projection MLP: 768 → 512
        self.projector = nn.Sequential(
            nn.Linear(CLIP_PROJ_DIM, PROJECTION_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # 5 task heads
        self.formal_head    = nn.Linear(PROJECTION_DIM, 1)
        self.season_head    = nn.Linear(PROJECTION_DIM, 4)
        self.gender_head    = nn.Linear(PROJECTION_DIM, 3)
        self.time_head      = nn.Linear(PROJECTION_DIM, 2)
        self.frequency_head = nn.Linear(PROJECTION_DIM, 2)

        # Loss functions
        self._mse = nn.MSELoss()
        self._ce  = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, 3, 224, 224) — pixel values pre-normalised with CLIP mean/std
        Returns:
            dict with keys: formal, season, gender, time, frequency
        """
        # CLIP vision model expects pixel_values kwarg
        vision_outputs = self.vision_model(pixel_values=x)
        # pooler_output is the [CLS] embedding, shape (B, hidden_dim)
        clip_feats = self.visual_proj(vision_outputs.pooler_output)   # (B, 768)

        proj = self.projector(clip_feats)   # (B, 512)

        return {
            "formal":    self.formal_head(proj).squeeze(1),   # (B,)
            "season":    self.season_head(proj),               # (B, 4)
            "gender":    self.gender_head(proj),               # (B, 3)
            "time":      self.time_head(proj),                 # (B, 2)
            "frequency": self.frequency_head(proj),            # (B, 2)
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
