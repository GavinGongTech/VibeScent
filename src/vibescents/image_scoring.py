from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from vibescents.fusion import min_max_normalize

logger = logging.getLogger(__name__)

SEASON_INDEX = {
    "spring": 0,
    "summer": 1,
    "fall": 2,
    "winter": 3,
}

GENDER_INDEX = {
    "male": 0,
    "female": 1,
    "neutral": 2,
}

FREQUENCY_INDEX = {
    "occasional": 0,
    "everyday": 1,
}

DEFAULT_HEAD_WEIGHTS = {
    "formal": 1.0,
    "season": 1.0,
    "time": 1.0,
    "gender": 1.0,
    "frequency": 1.0,
}

_EPSILON = 1e-8
_HEAD_NAMES = ("formal", "season", "time", "gender", "frequency")


@dataclass(frozen=True)
class ImageHeadProbabilities:
    formal: np.ndarray  # [casual, smart-casual, formal]
    season: np.ndarray  # [spring, summer, fall, winter]
    time: np.ndarray  # [day, night]
    gender: np.ndarray  # [male, female, neutral]
    frequency: np.ndarray  # [occasional, everyday]

    def as_dict(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in _HEAD_NAMES}


def discretize_formality(formality: float) -> int:
    if formality < 0.33:
        return 0
    if formality < 0.67:
        return 1
    return 2


def discretize_day_night(day_night: float) -> int:
    return 1 if day_night >= 0.5 else 0


def discretize_gender(gender: str | None) -> int:
    return GENDER_INDEX.get(
        (gender or "neutral").strip().lower(), GENDER_INDEX["neutral"]
    )


def discretize_frequency(frequency: str | None) -> int:
    return FREQUENCY_INDEX.get(
        (frequency or "everyday").strip().lower(), FREQUENCY_INDEX["everyday"]
    )


def season_target_index(likely_season: str | None, season_probs: np.ndarray) -> int:
    if likely_season is None:
        raise ValueError("likely_season is required for image scoring.")
    normalized = likely_season.strip().lower()
    if normalized == "all-season":
        return int(np.argmax(season_probs))
    if normalized not in SEASON_INDEX:
        raise ValueError(f"Unsupported likely_season: {likely_season!r}")
    return SEASON_INDEX[normalized]


def image_negative_log_likelihood(
    head_probabilities: Mapping[str, np.ndarray],
    fragrance: Mapping[str, Any],
    *,
    head_weights: Mapping[str, float] | None = None,
) -> float:
    for key in _HEAD_NAMES:
        if key not in head_probabilities:
            raise ValueError(f"Missing required head probabilities: {key}")

    weights = dict(head_weights or DEFAULT_HEAD_WEIGHTS)
    for key in _HEAD_NAMES:
        if key not in weights:
            raise ValueError(f"Missing head weight for: {key}")

    formality = _coerce_float(fragrance.get("formality"), "formality")
    day_night = _coerce_float(fragrance.get("day_night"), "day_night")
    likely_season = _coerce_str(fragrance.get("likely_season"), "likely_season")

    formal_target = discretize_formality(formality)
    time_target = discretize_day_night(day_night)
    season_target = season_target_index(likely_season, head_probabilities["season"])
    gender_target = discretize_gender(str(fragrance.get("gender", "neutral")))
    frequency_target = discretize_frequency(str(fragrance.get("frequency", "everyday")))

    nll = (
        weights["formal"]
        * np.log(_safe_probability(head_probabilities["formal"], formal_target))
        + weights["season"]
        * np.log(_safe_probability(head_probabilities["season"], season_target))
        + weights["time"]
        * np.log(_safe_probability(head_probabilities["time"], time_target))
        + weights["gender"]
        * np.log(_safe_probability(head_probabilities["gender"], gender_target))
        + weights["frequency"]
        * np.log(_safe_probability(head_probabilities["frequency"], frequency_target))
    )
    return float(-nll)


def image_score_likelihood(
    head_probabilities: Mapping[str, np.ndarray],
    fragrance: Mapping[str, Any],
    *,
    head_weights: Mapping[str, float] | None = None,
) -> float:
    nll = image_negative_log_likelihood(
        head_probabilities=head_probabilities,
        fragrance=fragrance,
        head_weights=head_weights,
    )
    return float(np.exp(-nll))


def score_candidate_pool(
    head_probabilities: Mapping[str, np.ndarray],
    candidates: Sequence[Mapping[str, Any]],
    *,
    head_weights: Mapping[str, float] | None = None,
) -> np.ndarray:
    if isinstance(head_probabilities, ImageHeadProbabilities):
        head_probabilities = head_probabilities.as_dict()
    if not candidates:
        return np.empty((0,), dtype=np.float32)
    likelihoods = np.array(
        [
            image_score_likelihood(
                head_probabilities=head_probabilities,
                fragrance=candidate,
                head_weights=head_weights,
            )
            for candidate in candidates
        ],
        dtype=np.float32,
    )
    return min_max_normalize(likelihoods)


class NeilCNNWrapper:
    """Wrap Neil's CNN-CLIP hybrid model with probability extraction helpers."""

    def __init__(self, model: Any, device: str = "cpu") -> None:
        self._model = model
        self._device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        model_builder: Callable[[], Any],
        device: str | None = None,
    ) -> "NeilCNNWrapper":
        import torch

        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = model_builder()
        state = torch.load(checkpoint_path, map_location=target_device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.to(target_device)
        model.eval()
        return cls(model=model, device=target_device)

    def predict_head_probabilities(self, image_tensor: Any) -> ImageHeadProbabilities:
        import torch

        with torch.no_grad():
            raw = self._model(image_tensor.to(self._device))
        outputs = _coerce_head_outputs(raw)
        return ImageHeadProbabilities(
            formal=_softmax_numpy(outputs["formal"]),
            season=_softmax_numpy(outputs["season"]),
            time=_softmax_numpy(outputs["time"]),
            gender=_softmax_numpy(outputs["gender"]),
            frequency=_softmax_numpy(outputs["frequency"]),
        )


def _coerce_head_outputs(raw: Any) -> dict[str, np.ndarray]:
    if isinstance(raw, Mapping):
        return {name: _to_numpy(raw[name]) for name in _HEAD_NAMES}
    if isinstance(raw, (tuple, list)) and len(raw) >= 5:
        return {name: _to_numpy(val) for name, val in zip(_HEAD_NAMES, raw[:5])}
    raise ValueError(
        "Unsupported CNN output format. Expected mapping or tuple/list with 5 heads."
    )


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value
    elif hasattr(value, "detach"):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    arr = arr.astype(np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D head output, got shape {arr.shape}.")
    return arr


def _softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    denom = np.sum(exp)
    if denom == 0:
        raise ValueError("Softmax denominator is zero.")
    return (exp / denom).astype(np.float32)


def _safe_probability(probabilities: np.ndarray, target_index: int) -> float:
    if target_index < 0 or target_index >= probabilities.shape[0]:
        raise ValueError(
            f"Target index {target_index} is out of bounds for shape {probabilities.shape}."
        )
    return float(np.clip(probabilities[target_index], _EPSILON, 1.0))


def _coerce_float(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required for image scoring.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc


def _coerce_str(value: Any, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} is required for image scoring.")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty.")
    return text


_CLIP_MODEL = "openai/clip-vit-base-patch32"

# Prompt bank — 3 prompts per class, ordered to match ImageHeadProbabilities indices:
# formal[0]=casual/[1]=smart-casual/[2]=formal; season[0]=spring/.../[3]=winter;
# time[0]=day/[1]=night; gender[0]=male/[1]=female/[2]=neutral;
# frequency[0]=occasional/[1]=everyday.
_CLIP_FORMAL_PROMPTS: list[list[str]] = [
    [
        "a person in ripped jeans and a graphic tee with sneakers",
        "someone in sweatpants and a hoodie running errands",
        "a relaxed weekend outfit with shorts and sandals",
    ],
    [
        "a person in chinos and a button down shirt with loafers",
        "a smart casual outfit with dark jeans and a blazer",
        "a neat polished look with trousers and a tucked in shirt",
    ],
    [
        "a person in a tailored suit and tie at a business meeting",
        "a woman in a floor length gown at a black tie event",
        "formal eveningwear including a tuxedo or cocktail dress",
    ],
]
_CLIP_SEASON_PROMPTS: list[list[str]] = [
    [
        "a light spring outfit with pastel colors and a thin jacket",
        "spring fashion featuring floral prints and transitional layers",
        "clothing suited for mild spring weather with light breathable fabrics",
    ],
    [
        "a summer outfit with shorts, sandals, and a lightweight top",
        "bright beach-ready summer wear for hot sunny weather",
        "minimal breathable summer clothing like sundresses or linen shirts",
    ],
    [
        "a fall outfit featuring warm earth tones, a cozy sweater, and boots",
        "autumn fashion with layered clothing, scarves, and plaid patterns",
        "transitional fall attire with a medium-weight jacket and warm colors",
    ],
    [
        "a winter outfit with a heavy coat, scarf, gloves, and warm layers",
        "cold-weather clothing including a wool coat and insulated boots",
        "bundled-up winter fashion for freezing temperatures and snow",
    ],
]
_CLIP_TIME_PROMPTS: list[list[str]] = [
    [
        "light-colored or pastel clothing in breathable natural fabrics like cotton or linen",
        "structured practical daywear such as chinos, a polo shirt, or a casual button-down",
        "low-key versatile outfit with subdued tones and comfortable everyday fabrics",
    ],
    [
        "a nighttime outfit for going out to bars, clubs, or evening events",
        "dark or glamorous evening wear suited for nightlife",
        "an outfit styled for after-dark occasions with bold or sleek aesthetics",
    ],
]
_CLIP_GENDER_PROMPTS: list[list[str]] = [
    [  # male
        "a man in a masculine outfit with tailored trousers and dress shoes",
        "a male presenting person in a structured jacket and button-down shirt",
        "menswear including suits, oxfords, or classic masculine attire",
    ],
    [  # female
        "a woman in a feminine outfit with a dress or skirt and heels",
        "a female presenting person in stylish women's fashion",
        "women's clothing including dresses, blouses, or feminine silhouettes",
    ],
    [  # neutral
        "a gender-neutral outfit with unisex clothing and no gendered elements",
        "an androgynous look with oversized silhouettes and neutral tones",
        "unisex streetwear that works equally well for any gender presentation",
    ],
]
_CLIP_FREQUENCY_PROMPTS: list[list[str]] = [
    [  # occasional
        "a special occasion outfit for a rare event like a wedding or gala",
        "formal attire reserved for once-in-a-while events and ceremonies",
        "a dressed-up look clearly intended for a special or infrequent occasion",
    ],
    [  # everyday
        "a casual everyday outfit suitable for regular daily activities",
        "relaxed everyday wear like jeans, t-shirts, or comfortable basics",
        "a practical low-key outfit for a regular day at work or running errands",
    ],
]


class CLIPImageScorer:
    """CLIP ViT-L/14 zero-shot classifier → ImageHeadProbabilities.

    Uses Neil's prompt bank (3 prompts per class). Per class, similarities are
    averaged across prompts before softmax — matching Neil's score_classification
    approach in backend/clip_zero_shot.py.
    """

    def __init__(self, model_name: str = _CLIP_MODEL) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        logger.info("CLIPImageScorer loading %s on %s", model_name, self._device)
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model = CLIPModel.from_pretrained(model_name).to(self._device).eval()
        self._torch = torch

        self._formal_embs = [self._encode_texts(p) for p in _CLIP_FORMAL_PROMPTS]
        self._season_embs = [self._encode_texts(p) for p in _CLIP_SEASON_PROMPTS]
        self._time_embs = [self._encode_texts(p) for p in _CLIP_TIME_PROMPTS]
        self._gender_embs = [self._encode_texts(p) for p in _CLIP_GENDER_PROMPTS]
        self._frequency_embs = [self._encode_texts(p) for p in _CLIP_FREQUENCY_PROMPTS]
        logger.info("CLIPImageScorer ready")

    def score_image(self, image_bytes: bytes) -> ImageHeadProbabilities:
        import torch.nn.functional as F

        img = (
            __import__("PIL.Image", fromlist=["Image"])
            .Image.open(BytesIO(image_bytes))
            .convert("RGB")
        )
        inputs = self._processor(images=img, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            img_emb = self._model.get_image_features(**inputs)
            img_emb = F.normalize(img_emb, dim=-1)  # (1, D)

        def _class_probs(class_emb_list: list) -> np.ndarray:
            sims = self._torch.stack(
                [(img_emb @ embs.T).mean(dim=-1) for embs in class_emb_list],
                dim=-1,
            ).squeeze(0)
            return F.softmax(sims, dim=-1).cpu().float().numpy()

        return ImageHeadProbabilities(
            formal=_class_probs(self._formal_embs),
            season=_class_probs(self._season_embs),
            time=_class_probs(self._time_embs),
            gender=_class_probs(self._gender_embs),
            frequency=_class_probs(self._frequency_embs),
        )

    def _encode_texts(self, prompts: list[str]):
        import torch.nn.functional as F

        inputs = self._processor(
            text=prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self._device)
        with self._torch.no_grad():
            embs = self._model.get_text_features(**inputs)
            return F.normalize(embs, dim=-1)  # (N, D)
