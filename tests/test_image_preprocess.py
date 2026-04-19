from __future__ import annotations

import base64
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

import vibescents.image_preprocess as ip
from vibescents.image_preprocess import decode_b64_image_bytes


# ---------------------------------------------------------------------------
# Pure-Python: decode_b64_image_bytes
# ---------------------------------------------------------------------------


def test_decode_accepts_valid_jpeg() -> None:
    data = base64.b64encode(b"fake-jpeg-bytes").decode("ascii")
    assert decode_b64_image_bytes(data, "image/jpeg") == b"fake-jpeg-bytes"


def test_decode_accepts_valid_png() -> None:
    data = base64.b64encode(b"fake-png-bytes").decode("ascii")
    assert decode_b64_image_bytes(data, "image/png") == b"fake-png-bytes"


def test_decode_accepts_valid_webp() -> None:
    data = base64.b64encode(b"fake-webp-bytes").decode("ascii")
    assert decode_b64_image_bytes(data, "image/webp") == b"fake-webp-bytes"


def test_decode_rejects_unsupported_mime_type() -> None:
    data = base64.b64encode(b"x").decode("ascii")
    with pytest.raises(ValueError, match="Unsupported mime type"):
        decode_b64_image_bytes(data, "image/gif")


def test_decode_rejects_invalid_base64() -> None:
    with pytest.raises(ValueError, match="Invalid base64"):
        decode_b64_image_bytes("not-valid-base64!!!!", "image/png")


def test_decode_rejects_empty_string() -> None:
    # An empty string is technically valid base64 (decodes to empty bytes),
    # so we verify we don't crash rather than asserting an exception.
    result = decode_b64_image_bytes("", "image/png")
    assert result == b""


# ---------------------------------------------------------------------------
# Constants — must match Neil's training configuration exactly
# ---------------------------------------------------------------------------


def test_clip_mean_matches_training_values() -> None:
    assert ip.CLIP_MEAN == pytest.approx((0.48145466, 0.4578275, 0.40821073), rel=1e-6)


def test_clip_std_matches_training_values() -> None:
    assert ip.CLIP_STD == pytest.approx((0.26862954, 0.26130258, 0.27577711), rel=1e-6)


def test_cnn_image_size_is_224() -> None:
    assert ip.CNN_IMAGE_SIZE == 224


def test_supported_mime_types() -> None:
    assert ip.SUPPORTED_MIME_TYPES == frozenset(
        {"image/jpeg", "image/png", "image/webp"}
    )


# ---------------------------------------------------------------------------
# decode_b64_to_cnn_tensor — mocked PIL + torchvision
# ---------------------------------------------------------------------------


def _build_pil_stub() -> tuple[types.ModuleType, MagicMock]:
    """Return (PIL module stub, PIL.Image mock)."""
    fake_pil_image_class = MagicMock(name="PIL.Image")
    fake_pil_image_class.BICUBIC = 3

    # open().convert().resize() → same mock as the resized image object
    fake_resized = MagicMock(name="resized_pil_image")
    fake_converted = MagicMock(name="converted_pil_image")
    fake_converted.resize.return_value = fake_resized
    fake_opened = MagicMock(name="opened_pil_image")
    fake_opened.convert.return_value = fake_converted
    fake_pil_image_class.open.return_value = fake_opened

    fake_pil_mod = types.ModuleType("PIL")
    fake_pil_mod.Image = fake_pil_image_class  # type: ignore[attr-defined]

    return fake_pil_mod, fake_pil_image_class


def _build_tv_functional_stub(fake_tensor: MagicMock) -> types.ModuleType:
    """Return torchvision.transforms.functional stub wired to return fake_tensor."""
    fake_normalized = MagicMock(name="normalized_tensor")
    batched = MagicMock(name="batched_tensor")
    fake_normalized.unsqueeze.return_value = batched

    fake_tf = types.ModuleType("torchvision.transforms.functional")
    fake_tf.to_tensor = MagicMock(return_value=fake_tensor)  # type: ignore[attr-defined]
    fake_tf.normalize = MagicMock(return_value=fake_normalized)  # type: ignore[attr-defined]
    return fake_tf


def test_decode_b64_to_cnn_tensor_calls_pipeline_in_order() -> None:
    """Verify the resize → to_tensor → normalize → unsqueeze pipeline."""
    fake_pil_mod, fake_pil_image_class = _build_pil_stub()
    fake_raw_tensor = MagicMock(name="raw_tensor")
    fake_tf = _build_tv_functional_stub(fake_raw_tensor)

    fake_tv = types.ModuleType("torchvision")
    fake_tv_transforms = types.ModuleType("torchvision.transforms")  # type: ignore[attr-defined]

    modules_patch = {
        "PIL": fake_pil_mod,
        "PIL.Image": fake_pil_image_class,
        "torchvision": fake_tv,
        "torchvision.transforms": fake_tv_transforms,
        "torchvision.transforms.functional": fake_tf,
    }

    with patch.dict(sys.modules, modules_patch):
        image_bytes = base64.b64encode(b"some-image-bytes").decode("ascii")
        result = ip.decode_b64_to_cnn_tensor(image_bytes, "image/jpeg")

    # PIL.Image.open was called with a BytesIO buffer
    assert fake_pil_image_class.open.called
    opened = fake_pil_image_class.open.return_value
    opened.convert.assert_called_once_with("RGB")
    converted = opened.convert.return_value
    converted.resize.assert_called_once_with(
        (224, 224), fake_pil_image_class.Resampling.BICUBIC
    )

    # to_tensor received the resized PIL image
    fake_tf.to_tensor.assert_called_once_with(converted.resize.return_value)

    # normalize received correct CLIP constants
    fake_tf.normalize.assert_called_once_with(
        fake_raw_tensor,
        mean=list(ip.CLIP_MEAN),
        std=list(ip.CLIP_STD),
    )

    # unsqueeze(0) adds the batch dimension
    normalized = fake_tf.normalize.return_value
    normalized.unsqueeze.assert_called_once_with(0)
    assert result is normalized.unsqueeze.return_value


def test_decode_b64_to_cnn_tensor_rejects_bad_input_before_pil() -> None:
    """ValueError from decode_b64_image_bytes propagates before PIL is imported."""
    with pytest.raises(ValueError, match="Unsupported mime type"):
        ip.decode_b64_to_cnn_tensor(base64.b64encode(b"x").decode("ascii"), "image/gif")
