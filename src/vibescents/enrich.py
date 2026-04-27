"""Enrich fragrance rows with structured vibe attributes and retrieval text."""

from __future__ import annotations

import json
import importlib
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
from pydantic import ValidationError

from vibescents.schemas import EnrichmentSchemaV2

# No API key needed — any HuggingFace instruct model works
# Alternatives: "Qwen/Qwen3-14B", "google/gemma-3-12b-it", "google/gemma-3-27b-it"
LOCAL_ENRICHMENT_MODEL = "Qwen/Qwen3-8B"
QWEN_ENRICHMENT_MODEL = LOCAL_ENRICHMENT_MODEL  # backward-compat alias
DEFAULT_BATCH_SIZE = 16
DELAY_BETWEEN_BATCHES = 1.0

ENRICHMENT_COLUMNS = list(EnrichmentSchemaV2.model_fields.keys())

SYSTEM_PROMPT = """\
You are a fragrance expert. Given a perfume's metadata, return a JSON object with EXACTLY these fields:

{
  "likely_season": "<one of: spring | summer | fall | winter | all-season>",
  "likely_occasion": "<short phrase, e.g. 'Evening gala' or 'Casual weekend'>",
  "formality": <float 0.0-1.0, where 0=casual, 1=black-tie>,
  "fresh_warm": <float 0.0-1.0, where 0=fresh/aquatic/citrus, 1=warm/oriental/musky>,
  "day_night": <float 0.0-1.0, where 0=daytime, 1=nighttime>,
  "gender": "<one of: male | female | neutral>",
  "frequency": "<one of: occasional | everyday>",
  "character_tags": ["<tag1>", "<tag2>", "<tag3>"],
  "vibe_sentence": "<one evocative sentence describing the overall vibe>",
  "longevity": "<short phrase, e.g. 'Long-lasting, 8+ hours' or 'Moderate, 4-6 hours'>",
  "projection": "<short phrase, e.g. 'Moderate sillage' or 'Intimate, skin-close'>",
  "mood_tags": ["<mood1>", "<mood2>"],
  "color_palette": ["<color1>", "<color2>"]
}

CRITICAL:
- Use ONLY the 13 field names shown above — no extra fields, no renamed fields
- character_tags must have 3–5 items; mood_tags and color_palette must have at least 1
- Do NOT use <think> tags or any chain-of-thought reasoning
- Start your response with '{' and end with '}' — no preamble, no explanation
"""


class EnrichmentClient(Protocol):
    def generate(self, prompt: str) -> EnrichmentSchemaV2:
        """Generate one enrichment object from a prompt."""

    def generate_batch(self, prompts: list[str]) -> list[EnrichmentSchemaV2 | None]:
        """Generate a batch of enrichment objects."""


@dataclass
class QwenOutlinesEnrichmentClient:
    model_name: str = QWEN_ENRICHMENT_MODEL

    def __post_init__(self) -> None:
        self._generator = _build_outlines_generator(self.model_name, EnrichmentSchemaV2)

    def generate(self, prompt: str) -> EnrichmentSchemaV2:
        # outlines generators are callable with the prompt
        raw = self._generator(f"{SYSTEM_PROMPT}\n\n{prompt}")
        parsed = _parse_enrichment(raw)
        if parsed is not None:
            return parsed
        repaired = _repair_payload(raw)
        parsed = _parse_enrichment(repaired)
        if parsed is not None:
            return parsed
        raise ValueError("Outlines output could not be parsed into EnrichmentSchemaV2.")

    def generate_batch(self, prompts: list[str]) -> list[EnrichmentSchemaV2 | None]:
        full_prompts = [f"{SYSTEM_PROMPT}\n\n{prompt}" for prompt in prompts]

        # outlines generators support batching by passing a list of prompts
        try:
            raw_outputs = self._generator(full_prompts)
        except Exception:
            # Fallback for older versions or if batching fails
            raw_outputs = [self._generator(p) for p in full_prompts]
        if not isinstance(raw_outputs, list):
            if isinstance(raw_outputs, tuple):
                raw_outputs = list(raw_outputs)
            else:
                raw_outputs = [raw_outputs]
        if len(raw_outputs) != len(full_prompts):
            raw_outputs = [self._generator(p) for p in full_prompts]

        results = []
        for raw in raw_outputs:
            parsed = _parse_enrichment(raw)
            if parsed is not None:
                results.append(parsed)
                continue
            repaired = _repair_payload(raw)
            parsed = _parse_enrichment(repaired)
            if parsed is not None:
                results.append(parsed)
                continue
            results.append(None)
        return results


@dataclass
class VLLMNativeEnrichmentClient:
    model_name: str = QWEN_ENRICHMENT_MODEL
    max_tokens: int = 4096
    # 0.0 = auto-detect: 0.90 on ≥70 GB VRAM (A100/Blackwell), 0.85 otherwise
    gpu_memory_utilization: float = 0.0

    def __post_init__(self) -> None:
        import contextlib

        from vllm import LLM, SamplingParams  # lazy import — vllm may not be installed
        from transformers import AutoTokenizer

        # suppress_stdout in vllm.distributed.parallel_state calls sys.stdout.fileno(),
        # which fails in any Jupyter context. Patch it to a no-op — it only suppresses
        # cosmetic NCCL output, so silencing the suppressor is safe.
        @contextlib.contextmanager
        def _noop_suppress():
            yield

        try:
            import vllm.utils.system_utils as _vsu

            _vsu.suppress_stdout = _noop_suppress
        except Exception:
            pass
        try:
            import vllm.distributed.parallel_state as _vps

            _vps.suppress_stdout = _noop_suppress
        except Exception:
            pass

        # Detect GPU VRAM via nvidia-smi — does NOT initialize CUDA, so vLLM can
        # keep fork mode. Calling any torch.cuda.* here would force vLLM into spawn
        # mode (CUDA already initialized), which hangs in Jupyter due to the
        # ipykernel stdout patch not being inherited by spawned subprocesses.
        import subprocess as _sp

        _vram_gb = 0.0
        try:
            _smi_out = _sp.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            if _smi_out:
                _vram_gb = int(_smi_out.split("\n")[0].strip()) / 1024.0
        except Exception:
            pass

        if self.gpu_memory_utilization == 0.0:
            if _vram_gb >= 90:  # Blackwell 6000 / H100
                self.gpu_memory_utilization = 0.92
            elif _vram_gb >= 70:  # A100 80 GB
                self.gpu_memory_utilization = 0.90
            else:  # T4 and smaller
                self.gpu_memory_utilization = 0.85

        _max_num_seqs = 512 if _vram_gb >= 70 else 256
        print(
            f"GPU VRAM: {_vram_gb:.0f} GB → util={self.gpu_memory_utilization}, "
            f"max_num_seqs={_max_num_seqs}"
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        schema_json = EnrichmentSchemaV2.model_json_schema()
        self._sampling_params = _build_vllm_sampling_params(
            SamplingParams,
            schema_json=schema_json,
            max_tokens=self.max_tokens,
        )

        print(f"Loading {self.model_name} via vLLM…")
        _llm_kwargs = dict(
            model=self.model_name,
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_num_seqs=_max_num_seqs,
        )
        try:
            self._llm = LLM(**_llm_kwargs, enable_prefix_caching=True)
        except TypeError:
            self._llm = LLM(**_llm_kwargs)

    def _apply_template(self, messages: list[dict]) -> str:
        """Apply chat template with thinking disabled to avoid wasting tokens."""
        kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
        # Qwen3 thinking models support enable_thinking=False to skip <think> blocks.
        # Without this, the model silently generates hundreds of think tokens before
        # the JSON, burning ~6x more compute than needed.
        try:
            return self._tokenizer.apply_chat_template(
                messages, **kwargs, enable_thinking=False
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(messages, **kwargs)

    def generate(self, prompt: str) -> EnrichmentSchemaV2:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        full_prompt = self._apply_template(messages)
        outputs = self._llm.generate([full_prompt], self._sampling_params)
        raw = outputs[0].outputs[0].text
        parsed = _parse_enrichment(raw)
        if parsed is not None:
            return parsed
        repaired = _repair_payload(raw)
        parsed = _parse_enrichment(repaired)
        if parsed is not None:
            return parsed
        raise ValueError(
            "vLLM native output could not be parsed into EnrichmentSchemaV2."
        )

    def generate_batch(self, prompts: list[str]) -> list[EnrichmentSchemaV2 | None]:
        full_prompts = []
        for p in prompts:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": p},
            ]
            full_prompts.append(self._apply_template(messages))
        outputs = self._llm.generate(full_prompts, self._sampling_params)
        results: list[EnrichmentSchemaV2 | None] = []
        for i, output in enumerate(outputs):
            raw = output.outputs[0].text
            parsed = _parse_enrichment(raw)
            if parsed is not None:
                results.append(parsed)
                continue

            # Debug first failure in batch
            if i == 0:
                print(f"\n[DEBUG] Parsing failed. Raw output start: {raw[:200]!r}")

            repaired = _repair_payload(raw)
            parsed = _parse_enrichment(repaired)
            if parsed is not None:
                results.append(parsed)
                continue
            results.append(None)
        return results


@dataclass
class GroqEnrichmentClient:
    """Free-tier Groq API client using Llama-3.1-8B with JSON schema structured output.

    Sign up at https://console.groq.com — no credit card required.
    Set GROQ_API_KEY environment variable before use.
    Free tier: ~6000 TPM → ~14 rows/min → 35k corpus in ~42 hours unattended.
    """

    model_name: str = "llama-3.1-8b-instant"
    requests_per_minute: int = 14
    max_retries: int = 6

    def __post_init__(self) -> None:
        import os

        self._api_key = os.environ.get("GROQ_API_KEY", "")
        if not self._api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Sign up free at https://console.groq.com, "
                "create an API key, then: export GROQ_API_KEY=your_key"
            )
        self._schema = EnrichmentSchemaV2.model_json_schema()
        self._min_interval = 60.0 / self.requests_per_minute
        self._last_request_time: float = 0.0

    def _call(self, prompt: str) -> str:
        import requests as _req
        import time

        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "EnrichmentSchemaV2",
                    "schema": self._schema,
                    "strict": True,
                },
            },
            "temperature": 0,
            "max_tokens": 1024,
        }

        backoff = 30
        for attempt in range(self.max_retries):
            resp = _req.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30,
            )
            self._last_request_time = time.time()
            if resp.status_code == 429:
                print(
                    f"[Groq] Rate limited — waiting {backoff}s (attempt {attempt + 1})"
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

        raise RuntimeError(f"Groq API failed after {self.max_retries} retries")

    def generate(self, prompt: str) -> EnrichmentSchemaV2:
        raw = self._call(prompt)
        parsed = _parse_enrichment(raw)
        if parsed is not None:
            return parsed
        repaired = _repair_payload(raw)
        parsed = _parse_enrichment(repaired)
        if parsed is not None:
            return parsed
        raise ValueError(f"Groq output unparseable: {raw[:200]}")

    def generate_batch(self, prompts: list[str]) -> list[EnrichmentSchemaV2 | None]:
        results: list[EnrichmentSchemaV2 | None] = []
        for prompt in prompts:
            try:
                results.append(self.generate(prompt))
            except Exception:
                results.append(None)
        return results


def _build_outlines_generator(model_name: str, schema: Any):
    try:
        import outlines
    except ImportError as exc:
        raise ImportError(
            "Local LLM enrichment requires outlines: pip install outlines transformers accelerate"
        ) from exc

    try:
        import vllm

        print(f"Attempting to load {model_name} via vLLM...")
        llm = vllm.LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=4096,  # Cap to prevent hang
            gpu_memory_utilization=0.85,
        )
        if hasattr(outlines, "from_vllm_offline"):
            model = outlines.from_vllm_offline(llm)
        elif hasattr(outlines.models, "VLLMOffline"):
            model = outlines.models.VLLMOffline(llm)
        elif hasattr(outlines.models, "vllm"):
            model = outlines.models.vllm(model_name)
        else:
            raise AttributeError(
                "No supported vLLM offline constructor found in outlines."
            )
    except Exception as e:
        print(f"vLLM load failed: {e}. Falling back to transformers (slower).")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Use torch_dtype="auto" to ensure we use the model's native precision (bf16/fp16)
        # instead of float32, which often causes VRAM overflow and CPU offloading.
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True
        )

        if hasattr(outlines, "from_transformers"):
            model = outlines.from_transformers(hf_model, tokenizer)
        elif hasattr(outlines.models, "Transformers"):
            model = outlines.models.Transformers(hf_model, tokenizer)
        else:
            try:
                # Some versions of outlines can take the hf_model directly
                model = outlines.models.transformers(hf_model, tokenizer)
            except Exception:
                # Fallback to model name
                model = outlines.models.transformers(
                    model_name, device="cuda", trust_remote_code=True
                )

    generator_api = getattr(outlines, "generator", None)
    if generator_api is not None:
        generator_json = getattr(generator_api, "json", None)
        if callable(generator_json):
            return generator_json(model, schema)

    generate_api = getattr(outlines, "generate", None)
    if generate_api is not None:
        generate_json = getattr(generate_api, "json", None)
        if callable(generate_json):
            return generate_json(model, schema)

    # Outlines >= 1.2 exposes a direct model call with output_type instead of
    # outlines.generate / outlines.generator helper modules.
    return _OutlinesStructuredAdapter(model=model, schema=schema)


def _flatten_schema(schema_json: dict[str, Any]) -> dict[str, Any]:
    """Inline $defs/$ref so vLLM's guided-decoding backends receive a flat schema.

    Pydantic emits $ref pointers for Literal types; xgrammar and some outlines
    versions reject schemas that contain unresolved references.
    """
    defs = schema_json.get("$defs", {})
    if not defs:
        return schema_json

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_name = node["$ref"].split("/")[-1]
                return _resolve(defs.get(ref_name, node))
            return {k: _resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    return _resolve(schema_json)


def _build_guided_decoding_params(schema_json: dict[str, Any]) -> Any | None:
    flat = _flatten_schema(schema_json)
    _last_err: list[str] = []

    for module_name, class_name in [
        ("vllm.sampling_params", "GuidedDecodingParams"),
        ("vllm.sampling_params", "GuidedDecodingConfig"),
        ("vllm", "GuidedDecodingParams"),
        ("vllm.entrypoints.openai.protocol", "GuidedDecodingParams"),
    ]:
        try:
            module = importlib.import_module(module_name)
            guided_cls = getattr(module, class_name)
        except (ImportError, AttributeError):
            continue

        for kwargs in (
            {"json": json.dumps(flat)},
            {"json": flat},
            {"json_schema": flat},
            {"schema": flat},
        ):
            try:
                return guided_cls(**kwargs)
            except Exception as exc:
                _last_err.append(f"{class_name}({list(kwargs)}): {exc}")
                continue

    if _last_err:
        print(f"[DEBUG] GuidedDecodingParams attempts failed: {_last_err[-1]}")
    return None


def _build_vllm_sampling_params(
    sampling_params_cls: Any,
    *,
    schema_json: dict[str, Any],
    max_tokens: int,
) -> Any:
    base_kwargs = {"max_tokens": max_tokens, "temperature": 0.0}

    # 1. Try modern vLLM approach (GuidedDecodingParams object)
    guided_decoding = _build_guided_decoding_params(schema_json)
    if guided_decoding is not None:
        try:
            return sampling_params_cls(**base_kwargs, guided_decoding=guided_decoding)
        except Exception as exc:
            print(f"[DEBUG] SamplingParams(guided_decoding=...) failed: {exc}")

    # 2. Try older vLLM approach (direct keywords in SamplingParams)
    for key in ["guided_json", "json", "guided_decoding_json"]:
        for val in [schema_json, json.dumps(schema_json)]:
            try:
                return sampling_params_cls(**base_kwargs, **{key: val})
            except Exception:
                continue

    print(
        "[WARN] vLLM guided decoding unavailable — using prompt-based schema enforcement instead."
    )
    return sampling_params_cls(**base_kwargs)


def _call_with_schema(
    fn: Any,
    prompt: Any,
    schema: Any,
    *,
    max_new_tokens: int,
    sampling_params: Any | None,
) -> Any:
    attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
        (
            (),
            {
                "output_type": schema,
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
            },
        ),
        ((schema,), {"max_new_tokens": max_new_tokens, "temperature": 0.0}),
    ]
    if sampling_params is not None:
        attempts.extend(
            [
                ((), {"output_type": schema, "sampling_params": sampling_params}),
                ((schema,), {"sampling_params": sampling_params}),
            ]
        )
    attempts.extend(
        [
            ((), {"output_type": schema}),
            ((schema,), {}),
            ((), {}),
        ]
    )

    last_type_error: TypeError | None = None
    for args, kwargs in attempts:
        try:
            return fn(prompt, *args, **kwargs)
        except TypeError as exc:
            last_type_error = exc
            continue
    if last_type_error is not None:
        raise last_type_error
    raise TypeError("Unable to call outlines generator with the current API.")


@dataclass
class _OutlinesStructuredAdapter:
    model: Any
    schema: Any
    max_new_tokens: int = 512

    def __post_init__(self) -> None:
        self._sampling_params = None
        try:
            from vllm import SamplingParams

            self._sampling_params = SamplingParams(
                max_tokens=self.max_new_tokens,
                temperature=0.0,
            )
        except Exception:
            self._sampling_params = None

    def __call__(self, prompts: str | list[str]) -> Any:
        if isinstance(prompts, list):
            batch_fn = getattr(self.model, "batch", None)
            if callable(batch_fn):
                try:
                    return _call_with_schema(
                        batch_fn,
                        prompts,
                        self.schema,
                        max_new_tokens=self.max_new_tokens,
                        sampling_params=self._sampling_params,
                    )
                except TypeError:
                    pass
            return [self._call_single(prompt) for prompt in prompts]
        return self._call_single(prompts)

    def _call_single(self, prompt: str) -> Any:
        return _call_with_schema(
            self.model,
            prompt,
            self.schema,
            max_new_tokens=self.max_new_tokens,
            sampling_params=self._sampling_params,
        )


def _repair_payload(payload: Any) -> Any:
    try:
        from json_repair import repair_json
    except ImportError:
        return payload
    return repair_json(str(payload))


def _parse_enrichment(payload: Any) -> EnrichmentSchemaV2 | None:
    try:
        if isinstance(payload, EnrichmentSchemaV2):
            return payload
        if isinstance(payload, dict):
            return EnrichmentSchemaV2.model_validate(payload)

        if isinstance(payload, str):
            # 1. Strip reasoning blocks - handles both closed and unclosed (cut off) blocks
            import re

            cleaned = re.sub(
                r"<think>.*?(?:</think>|$)", "", payload, flags=re.DOTALL
            ).strip()

            # 2. Try direct parse
            try:
                parsed = json.loads(cleaned)
                return EnrichmentSchemaV2.model_validate(parsed)
            except json.JSONDecodeError:
                pass

            # 3. Extract JSON object using regex if model yapped around it
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    return EnrichmentSchemaV2.model_validate(parsed)
                except json.JSONDecodeError:
                    pass

        return None
    except (ValidationError, TypeError, ValueError):
        return None


def _build_prompt(row: pd.Series) -> str:
    parts = [f"Name: {row.get('name', '')}"]
    for column, label in [
        ("brand", "Brand"),
        ("top_notes", "Top notes"),
        ("middle_notes", "Middle notes"),
        ("base_notes", "Base notes"),
        ("main_accords", "Accords"),
        ("gender", "Gender"),
        ("concentration", "Concentration"),
        ("category", "Category"),
    ]:
        value = row.get(column)
        if pd.notna(value):
            parts.append(f"{label}: {value}")

    parts.append(
        "\nReturn ONLY a JSON object with these exact field names: "
        "likely_season, likely_occasion, formality, fresh_warm, day_night, "
        "gender, frequency, character_tags, vibe_sentence, longevity, projection, "
        "mood_tags, color_palette. No other fields. No thinking tags."
    )
    return "\n".join(parts)


def _shrink_prompt(prompt: str, factor: float = 0.7) -> str:
    clipped = prompt[: max(1, int(len(prompt) * factor))]
    return clipped if clipped.endswith("\n") else f"{clipped}\n"


def _serialize_value(value: Any) -> Any:
    if isinstance(value, list):
        return json.dumps(value)
    return value


def _append_failure_record(
    failures_path: str | None,
    *,
    row_index: int,
    prompt: str,
    error: str,
) -> None:
    if failures_path is None:
        return
    record = {
        "row_index": row_index,
        "error": error,
        "prompt": prompt,
    }
    with open(failures_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _resolve_client(*, model_name: str | None) -> EnrichmentClient:
    effective = model_name or LOCAL_ENRICHMENT_MODEL
    # 1. GPU: vLLM native guided decoding (fastest)
    try:
        return VLLMNativeEnrichmentClient(model_name=effective)
    except Exception:
        pass
    # 2. Free API: Groq (CPU-friendly, ~42h for 35k rows, GROQ_API_KEY required)
    try:
        return GroqEnrichmentClient()
    except EnvironmentError:
        pass
    # 3. Local CPU fallback: Outlines + transformers (slow)
    return QwenOutlinesEnrichmentClient(model_name=effective)


def enrich_dataframe(
    df: pd.DataFrame,
    *,
    model_name: str | None = None,
    client: EnrichmentClient | None = None,
    provider: str = "local",  # kept for backward-compat, always uses local LLM
    max_rows: int | None = None,
    resume_from: int = 0,
    checkpoint_path: str | None = None,
    failures_path: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    """Enrich dataframe rows with EnrichmentSchemaV2 fields."""
    client = client or _resolve_client(model_name=model_name)

    work = df.copy()
    for column in ENRICHMENT_COLUMNS:
        if column not in work.columns:
            work[column] = pd.Series([None] * len(work), dtype="object")

    end = min(len(work), resume_from + max_rows) if max_rows else len(work)
    subset = work.iloc[resume_from:end]

    total = len(subset)
    processed = 0
    failed = 0

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = subset.iloc[batch_start:batch_end]

        prompts = [_build_prompt(row) for _, row in batch.iterrows()]
        indices = batch.index.tolist()

        try:
            records = client.generate_batch(prompts)
            for i, record in enumerate(records):
                idx = indices[i]
                if record is None:
                    # Retry single row with shrunken prompt
                    try:
                        record = client.generate(_shrink_prompt(prompts[i]))
                    except Exception as retry_exc:
                        failed += 1
                        _append_failure_record(
                            failures_path,
                            row_index=int(idx),
                            prompt=prompts[i],
                            error=f"Batch fail; retry={retry_exc}",
                        )
                        continue

                for column in ENRICHMENT_COLUMNS:
                    work.at[idx, column] = _serialize_value(getattr(record, column))
                processed += 1
        except Exception as batch_exc:
            print(f"Batch processing failed: {batch_exc}. Falling back to row-by-row.")
            for idx, row in batch.iterrows():
                prompt = _build_prompt(row)
                try:
                    record = client.generate(prompt)
                except Exception as exc:
                    try:
                        record = client.generate(_shrink_prompt(prompt))
                    except Exception as retry_exc:
                        failed += 1
                        _append_failure_record(
                            failures_path,
                            row_index=int(idx),
                            prompt=prompt,
                            error=f"{exc}; retry={retry_exc}",
                        )
                        continue

                for column in ENRICHMENT_COLUMNS:
                    work.at[idx, column] = _serialize_value(getattr(record, column))
                processed += 1

        done = min(batch_end, total)
        print(f"  [{done}/{total}] processed={processed} failed={failed}")
        if checkpoint_path:
            work.to_csv(checkpoint_path, index=False)
        if batch_end < total:
            time.sleep(DELAY_BETWEEN_BATCHES)

    print(f"\nEnrichment complete: {processed} ok, {failed} failed out of {total}")
    return work


def _build_retrieval_text(row: pd.Series) -> str:
    parts: list[str] = []

    name = row.get("name", "")
    brand = row.get("brand", "")
    if pd.notna(brand) and brand:
        parts.append(f"Brand: {brand} | Name: {name}")
    else:
        parts.append(f"Name: {name}")

    if pd.notna(row.get("main_accords")):
        parts.append(f"Accords: {row['main_accords']}")

    note_parts = []
    for column, label in [
        ("top_notes", "Top"),
        ("middle_notes", "Heart"),
        ("base_notes", "Base"),
    ]:
        value = row.get(column)
        if pd.notna(value):
            note_parts.append(f"{label}: {value}")
    if note_parts:
        parts.append(" | ".join(note_parts))

    if pd.notna(row.get("likely_season")):
        parts.append(f"Season: {row['likely_season']}")
    if pd.notna(row.get("likely_occasion")):
        parts.append(f"Best for: {row['likely_occasion']}")
    if pd.notna(row.get("formality")):
        formality = float(row["formality"])
        level = "low" if formality < 0.33 else "medium" if formality < 0.67 else "high"
        parts.append(f"Formality: {level}")

    for field_name, label in [
        ("character_tags", "Character"),
        ("mood_tags", "Mood"),
        ("color_palette", "Palette"),
    ]:
        value = row.get(field_name)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        if isinstance(value, str) and value:
            parts.append(f"{label}: {value}")
        elif isinstance(value, list) and value:
            parts.append(f"{label}: {', '.join(value)}")

    if pd.notna(row.get("longevity")):
        parts.append(f"Longevity: {row['longevity']}")
    if pd.notna(row.get("projection")):
        parts.append(f"Projection: {row['projection']}")
    if pd.notna(row.get("vibe_sentence")):
        parts.append(f"Vibe: {row['vibe_sentence']}")

    return " | ".join(parts)


def build_retrieval_text(df: pd.DataFrame) -> pd.DataFrame:
    """Build retrieval_text from raw and enriched fields."""
    work = df.copy()

    def _safe_parse_json_array(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("["):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    saved_columns: dict[str, pd.Series] = {}
    for column in ("character_tags", "mood_tags", "color_palette"):
        if column in work.columns:
            saved_columns[column] = work[column].copy()
            work[column] = work[column].apply(_safe_parse_json_array)

    work["retrieval_text"] = work.apply(_build_retrieval_text, axis=1)
    for column, saved in saved_columns.items():
        work[column] = saved
    return work


def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Enrich fragrance dataset with vibe attributes"
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--resume-from", type=int, default=0)
    parser.add_argument(
        "--model",
        default=LOCAL_ENRICHMENT_MODEL,
        help="HF model: Qwen/Qwen3-8B, google/gemma-3-12b-it, etc.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--failures-log", default=None, help="Optional JSONL path for failed rows."
    )
    args = parser.parse_args()

    checkpoint_path = args.output_csv + ".ckpt"
    failures_path = args.failures_log or args.output_csv + ".failures.jsonl"

    print(f"Loading {args.input_csv}...")
    df = pd.read_csv(args.input_csv)

    if args.resume_from > 0 and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint {checkpoint_path}...")
        df_ckpt = pd.read_csv(checkpoint_path)
        for column in df_ckpt.columns:
            if column not in df.columns:
                df[column] = df_ckpt[column]
        df.update(df_ckpt)

    if "fragrance_id" not in df.columns:
        df.insert(0, "fragrance_id", df.index.astype(str))
        print(f"Added fragrance_id column (0 to {len(df) - 1})")

    print(
        f"Enriching rows starting at index {args.resume_from} "
        f"with model={args.model}..."
    )
    enriched = enrich_dataframe(
        df,
        model_name=args.model,
        max_rows=args.max_rows,
        resume_from=args.resume_from,
        checkpoint_path=checkpoint_path,
        failures_path=failures_path,
        batch_size=args.batch_size,
    )
    enriched = build_retrieval_text(enriched)
    enriched.to_csv(args.output_csv, index=False)
    print(f"Saved {args.output_csv} ({enriched.shape[0]} rows).")


if __name__ == "__main__":
    main()
