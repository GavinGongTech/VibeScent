from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

# Build _stubs dict
_stubs = {}

# torch
torch_stub = MagicMock()
torch_stub.Tensor = object
torch_stub.FloatTensor = object
torch_stub.LongTensor = object
torch_stub.no_grad = MagicMock
torch_stub.device = MagicMock
torch_stub.arange = MagicMock
torch_stub.bfloat16 = "bfloat16"

# torch_stub.cuda as a module with is_available = lambda: False
cuda_stub = types.ModuleType("torch.cuda")
cuda_stub.is_available = lambda: False
torch_stub.cuda = cuda_stub

# torch_stub.nn as a MagicMock with nn.functional.normalize = MagicMock
torch_nn_stub = MagicMock()
torch_nn_functional_stub = MagicMock()
torch_nn_functional_stub.normalize = MagicMock()
torch_nn_stub.functional = torch_nn_functional_stub
torch_stub.nn = torch_nn_stub

_stubs["torch"] = torch_stub
_stubs["torch.cuda"] = cuda_stub
_stubs["torch.nn"] = torch_nn_stub
_stubs["torch.nn.functional"] = torch_nn_functional_stub

# PIL
pil_stub = MagicMock()
pil_image_stub = MagicMock()
pil_image_stub.Image = object
pil_stub.Image = pil_image_stub
_stubs["PIL"] = pil_stub
_stubs["PIL.Image"] = pil_image_stub

# transformers
_stubs["transformers"] = MagicMock()

cache_utils_stub = MagicMock()
cache_utils_stub.Cache = object
_stubs["transformers.cache_utils"] = cache_utils_stub

modeling_outputs_stub = MagicMock()
modeling_outputs_stub.ModelOutput = object
_stubs["transformers.modeling_outputs"] = modeling_outputs_stub

modeling_qwen3_vl_stub = MagicMock()
modeling_qwen3_vl_stub.Qwen3VLConfig = MagicMock()
modeling_qwen3_vl_stub.Qwen3VLModel = MagicMock()
modeling_qwen3_vl_stub.Qwen3VLPreTrainedModel = object
_stubs["transformers.models.qwen3_vl.modeling_qwen3_vl"] = modeling_qwen3_vl_stub

processing_qwen3_vl_stub = MagicMock()
processing_qwen3_vl_stub.Qwen3VLProcessor = MagicMock()
_stubs["transformers.models.qwen3_vl.processing_qwen3_vl"] = processing_qwen3_vl_stub

processing_utils_stub = MagicMock()
processing_utils_stub.Unpack = MagicMock()
_stubs["transformers.processing_utils"] = processing_utils_stub

utils_stub = MagicMock()
utils_stub.TransformersKwargs = dict
_stubs["transformers.utils"] = utils_stub

# qwen_vl_utils
_stubs["qwen_vl_utils"] = MagicMock()
vision_process_stub = MagicMock()
vision_process_stub.process_vision_info = MagicMock()
_stubs["qwen_vl_utils.vision_process"] = vision_process_stub

# import sample_frames with stubs active, then restore
with patch.dict(sys.modules, _stubs):
    if "vibescents.qwen3_vl_embedding" in sys.modules:
        del sys.modules["vibescents.qwen3_vl_embedding"]
    from vibescents.qwen3_vl_embedding import sample_frames


def test_sample_frames_exact_count():
    """10 frames, num_segments=5, max_segments=10 → returns exactly 5 frames from correct positions"""
    frames = list(range(10))
    # np.linspace(0, 9, 5, dtype=int) -> [0, 2, 4, 6, 9]
    result = sample_frames(frames, num_segments=5, max_segments=10)
    assert result == [0, 2, 4, 6, 9]
    assert len(result) == 5


def test_sample_frames_respects_max_segments():
    """10 frames, num_segments=10, max_segments=3 → returns exactly 3 frames"""
    frames = list(range(10))
    # np.linspace(0, 9, 10, dtype=int) -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    result = sample_frames(frames, num_segments=10, max_segments=3)
    assert result == [0, 1, 2]
    assert len(result) == 3


def test_sample_frames_pads_short_list():
    """2 frames, num_segments=5, max_segments=5 → returns 5 items (pads with last frame)"""
    frames = [10, 20]
    # np.linspace(0, 1, 5, dtype=int) -> [0, 0, 0, 0, 1]
    result = sample_frames(frames, num_segments=5, max_segments=5)
    assert result == [10, 10, 10, 10, 20]
    assert len(result) == 5


def test_sample_frames_single_frame():
    """1 frame, num_segments=3, max_segments=3 → returns 3 copies of that frame"""
    frames = ["frame1"]
    # np.linspace(0, 0, 3, dtype=int) -> [0, 0, 0]
    result = sample_frames(frames, num_segments=3, max_segments=3)
    assert result == ["frame1", "frame1", "frame1"]
    assert len(result) == 3
