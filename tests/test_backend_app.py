from __future__ import annotations

import base64

import pytest

from vibescents.backend_app import decode_request_image
from vibescents.schemas import ContextInput, RecommendRequest


def test_decode_request_image_accepts_valid_base64() -> None:
    payload = RecommendRequest(
        image=base64.b64encode(b"hello").decode("ascii"),
        mimeType="image/png",
        context=ContextInput(),
    )
    assert decode_request_image(payload) == b"hello"


def test_decode_request_image_rejects_invalid_base64() -> None:
    payload = RecommendRequest(
        image="not-base64",
        mimeType="image/png",
        context=ContextInput(),
    )
    with pytest.raises(ValueError):
        decode_request_image(payload)
