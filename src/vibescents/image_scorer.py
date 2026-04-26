from __future__ import annotations

# Lazy-import torch/transformers so the module can be imported without GPU.
# SigLIP2ImageScorer.__init__ does the actual loading.

import logging
from io import BytesIO

import numpy as np

from vibescents.image_scoring import ImageHeadProbabilities

logger = logging.getLogger(__name__)

_SIGLIP2_MODEL = "google/siglip2-base-patch16-224"

# Text prompts per classification head — ordered to match index expectations in
# ImageHeadProbabilities: formal[0]=low, [1]=mid, [2]=high; season[0..3]=spring/summer/fall/winter; time[0]=day, [1]=night
_FORMAL_PROMPTS = [
    "a person wearing casual everyday clothes",
    "a person wearing smart casual business attire",
    "a person wearing elegant formal wear",
]
_SEASON_PROMPTS = [
    "a spring outfit with light pastel colors and floral patterns",
    "a summer outfit with bright colors and light fabrics",
    "a fall outfit with warm earthy tones and layered clothing",
    "a winter outfit with dark colors and heavy fabrics",
]
_TIME_PROMPTS = [
    "a daytime outfit appropriate for morning or afternoon",
    "a nighttime outfit appropriate for evening or late night",
]


class SigLIP2ImageScorer:
    """CPU-viable SigLIP 2 zero-shot classifier → ImageHeadProbabilities.

    SigLIP 2 (google/siglip2-base-patch16-224, Feb 2025) uses sigmoid loss during
    training, making each (image, text) pair independently scored. For our multi-class
    heads we still apply softmax over candidate logits — appropriate when exactly one
    class wins. We use the model's own logit_scale and logit_bias instead of a
    hardcoded temperature constant.
    """

    def __init__(self, model_name: str = _SIGLIP2_MODEL) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self._device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        logger.info("SigLIP2ImageScorer loading %s on %s", model_name, self._device)
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self._device).eval()
        self._torch = torch

        # Extract trained temperature parameters (logit_scale, logit_bias)
        self._logit_scale = self._model.logit_scale.exp().item()
        self._logit_bias = (
            self._model.logit_bias.item()
            if hasattr(self._model, "logit_bias") and self._model.logit_bias is not None
            else 0.0
        )

        # Pre-encode all text prompts once
        self._formal_embs = self._encode_texts(_FORMAL_PROMPTS)   # (3, D)
        self._season_embs = self._encode_texts(_SEASON_PROMPTS)   # (4, D)
        self._time_embs   = self._encode_texts(_TIME_PROMPTS)     # (2, D)
        logger.info("SigLIP2ImageScorer ready (logit_scale=%.2f, logit_bias=%.4f)",
                    self._logit_scale, self._logit_bias)

    def score_image(self, image_bytes: bytes) -> ImageHeadProbabilities:
        """Run SigLIP 2 zero-shot on raw image bytes → ImageHeadProbabilities."""
        import torch.nn.functional as F

        img = __import__("PIL.Image", fromlist=["Image"]).Image.open(BytesIO(image_bytes)).convert("RGB")
        inputs = self._processor(images=[img], return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            img_emb = self._model.get_image_features(**inputs)  # (1, D)
            img_emb = F.normalize(img_emb, dim=-1)

        def _probs(text_embs) -> np.ndarray:
            sims = (img_emb @ text_embs.T).squeeze(0)          # (N,)
            logits = self._logit_scale * sims + self._logit_bias
            return F.softmax(logits, dim=-1).cpu().float().numpy()

        return ImageHeadProbabilities(
            formal=_probs(self._formal_embs),
            season=_probs(self._season_embs),
            time=_probs(self._time_embs),
        )

    def _encode_texts(self, prompts: list[str]) -> np.ndarray:
        import torch.nn.functional as F
        inputs = self._processor(text=prompts, return_tensors="pt", padding="max_length", truncation=True).to(self._device)
        with self._torch.no_grad():
            embs = self._model.get_text_features(**inputs)
            return F.normalize(embs, dim=-1)  # (N, D)
