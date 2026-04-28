"""Image preprocessing for Neil's CNN-CLIP hybrid model.

Decodes base64-encoded images and prepares normalized tensors for CNN inference.
Heavy imports (PIL, torch, torchvision) are guarded inside function bodies so
the module loads on CPU-only machines for testing and linting.
"""

from __future__ import annotations

import base64
import binascii
import io
from typing import Any

# CLIP normalization constants used during training of Neil's CNN-CLIP hybrid.
# These must NOT be changed — they are baked into the checkpoint weights.
CLIP_MEAN: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

# Pre-converted list forms for torchvision.transforms.functional.normalize.
_CLIP_MEAN_LIST: list[float] = list(CLIP_MEAN)
_CLIP_STD_LIST: list[float] = list(CLIP_STD)

# Spatial resolution expected by both the ResNet-50 backbone and the CLIP ViT-L/14.
CNN_IMAGE_SIZE: int = 224

SUPPORTED_MIME_TYPES: frozenset[str] = frozenset(
    {"image/jpeg", "image/png", "image/webp"}
)


def decode_b64_image_bytes(image_b64: str, mime_type: str) -> bytes:
    """Decode a base64 image string to raw bytes.

    Parameters
    ----------
    image_b64:
        Base64-encoded image string (no ``data:`` URL prefix).
    mime_type:
        MIME type of the image — must be one of
        ``"image/jpeg"``, ``"image/png"``, or ``"image/webp"``.

    Raises
    ------
    ValueError
        If *mime_type* is not supported or *image_b64* is not valid base64.
    """
    if mime_type not in SUPPORTED_MIME_TYPES:
        raise ValueError(
            f"Unsupported mime type {mime_type!r}. "
            f"Expected one of {sorted(SUPPORTED_MIME_TYPES)}."
        )
    try:
        return base64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image payload.") from exc


def decode_b64_to_cnn_tensor(image_b64: str, mime_type: str) -> Any:
    """Decode a base64 image and return a CLIP-normalized CNN input tensor.

    Performs the following steps:
    1. Validate mime type and decode base64 to bytes.
    2. Open with PIL and convert to RGB.
    3. Resize to ``(224, 224)`` with bicubic interpolation.
    4. Convert to ``(3, 224, 224)`` float32 tensor in ``[0, 1]``.
    5. Apply CLIP normalization (``mean = CLIP_MEAN``, ``std = CLIP_STD``).
    6. Add batch dimension to produce shape ``(1, 3, 224, 224)``.

    Parameters
    ----------
    image_b64:
        Base64-encoded image string (no ``data:`` URL prefix).
    mime_type:
        MIME type of the image.

    Returns
    -------
    torch.Tensor
        Shape ``(1, 3, 224, 224)``, dtype float32, CLIP-normalized.
        Ready to pass directly to Neil's ``CNNCLIPHybrid`` forward pass.
    """
    # Validate and decode before importing heavy deps — keeps the error fast and testable.
    image_bytes = decode_b64_image_bytes(image_b64, mime_type)

    from PIL import Image  # noqa: PLC0415
    import torchvision.transforms.functional as TF  # noqa: PLC0415

    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pil_image = pil_image.resize(
        (CNN_IMAGE_SIZE, CNN_IMAGE_SIZE), Image.Resampling.BICUBIC
    )

    # to_tensor: PIL (H, W, C) uint8 → float32 Tensor (C, H, W) in [0, 1]
    tensor = TF.to_tensor(pil_image)
    tensor = TF.normalize(tensor, mean=_CLIP_MEAN_LIST, std=_CLIP_STD_LIST)
    return tensor.unsqueeze(0)  # (1, 3, 224, 224)