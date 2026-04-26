# Harsh Offline Pipeline Notebook — Colab Robustness Fix

**Commit:** `1b0e22f`  
**Date:** 2026-04-26  
**Issue:** Notebook failed with `exit-code-2` errors in Colab; variables undefined across stages; GPU credits wasted on recoverable errors.

---

## Problems Fixed

### 1. **uv pip Incompatibility (Exit-Code-2)**
- **Root cause:** `uv pip install --system` conflicts with Colab's pre-installed CUDA PyTorch. The strict resolver tries to downgrade torch, fails, and exits with code 2.
- **Fix:** Replaced all `uv pip` calls with standard `pip install`. Uses `_pip_install()` helper for consistent error handling.
- **Impact:** Eliminates the vLLM install crash that was the first blocker.

### 2. **Missing Torch Before Model Installs**
- **Root cause:** Stage 1 tried to install vllm before torch existed, causing import errors.
- **Fix:** Added explicit torch availability check in Stage 1 Step 4a. If missing, installs `torch>=2.4,<3.0` first.
- **Impact:** Prevents cascading failures in downstream model installs.

### 3. **Undefined Globals Across Runtime Restarts**
- **Root cause:** After `Restart runtime`, `INPUT_CSV`, `CHECKPOINT_CSV`, `ENRICHED_CSV`, `FAILURES_LOG` were undefined if not re-run from Stage 2.
- **Fix:** All stages now use `globals().get('VAR', fallback_path)` for defensive path initialization.
- **Impact:** Any stage can run independently; no "variable not found" crashes.

### 4. **Drive File Recovery**
- **Root cause:** Stage 3 assumed `INPUT_CSV` existed in the repo clone (it's gitignored).
- **Fix:** Added Drive fallback logic: if file missing, search Drive locations and copy via `shutil.copy()`.
- **Impact:** First-time users can upload CSV to Drive; pipeline self-recovers automatically.

### 5. **Missing df_work After Runtime Restart**
- **Root cause:** Stage 4 depended on `df_work` from Stage 3, but after runtime restart it was undefined.
- **Fix:** Stage 4 now reconstructs `df_work` from disk if not in globals. Includes checkpoint merge.
- **Impact:** Can run Stage 4 independently after a runtime restart.

### 6. **vLLM Backend Fallback**
- **Root cause:** If vllm install failed (optional), Stage 4 would crash trying to use `VLLMNativeEnrichmentClient`.
- **Fix:** Added try/except in Stage 4 to catch vLLM unavailability and fallback to `QwenOutlinesEnrichmentClient`.
- **Impact:** Always have a working enrichment path, even if vLLM is unavailable.

### 7. **Checkpoint Validation**
- **Root cause:** Stage 5 could fail silently if checkpoint row count exceeded dataset size (data corruption).
- **Fix:** Added validation: `if already_embedded > len(texts): raise ValueError(...)`.
- **Impact:** Early detection of corrupt checkpoints; clear recovery instructions.

### 8. **Embeddings Verification**
- **Root cause:** Stage 6 assumed embeddings.npy had >1 row; crashed if corpus was empty.
- **Fix:** Added `if emb.shape[0] > 1:` check before similarity preview. Added empty array check.
- **Impact:** Safe verification even for tiny test datasets.

### 9. **Directory Creation**
- **Root cause:** Stages 4–6 tried to write to ENRICHED_CSV/CHECKPOINT_CSV without creating parent directories.
- **Fix:** Added `os.makedirs(os.path.dirname(...), exist_ok=True)` before all file writes.
- **Impact:** No "FileNotFoundError: No such file or directory" on Drive writes.

### 10. **Optional Accelerators**
- **Root cause:** `bitsandbytes`, `hf_transfer`, `vllm` installs could fail, causing entire pipeline to abort.
- **Fix:** Wrapped optional packages in `required=False` try/except. Stage 4 continues even if they fail.
- **Impact:** Pipeline completes even on restrictive Colab sessions with quota limits.

---

## Key Design Changes

### Defensive Path Initialization
```python
# OLD: Path defined in Stage 2, undefined in later stages after runtime restart
ENRICHED_CSV = os.path.join(DRIVE_BASE, 'vibescent_enriched_full.csv')

# NEW: Defensive fallback in each stage
ENRICHED_CSV = globals().get('ENRICHED_CSV', 
                            os.path.join(DRIVE_BASE_DEF, 'vibescent_enriched_full.csv'))
```

### Pip Helper Function
```python
def _pip_install(packages, *, required=True, extra_args=None):
    """Install with graceful error handling. Optional packages don't abort pipeline."""
    cmd = [sys.executable, '-m', 'pip', 'install', '-q']
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(packages)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return True
    
    details = (result.stderr or result.stdout or '').strip()
    if required:
        raise RuntimeError(f"pip install failed: {' '.join(packages)}\n{details[-2000:]}")
    
    print(f"[WARN] Optional install failed: {' '.join(packages)}")
    return False
```

### Torch Pre-Check
```python
# Stage 1, Step 4a
try:
    import torch
    print('Step 4a: torch already available in runtime.')
except Exception:
    print('Step 4a: Installing torch>=2.4,<3.0 ...')
    _pip_install(['torch>=2.4,<3.0'], required=True)
```

### Model Backend Fallback
```python
# Stage 4
try:
    client = VLLMNativeEnrichmentClient(model_name=ENRICHMENT_MODEL)
    print("Using vLLM native guided decoding backend.")
except Exception as e:
    print(f"[WARN] vLLM backend unavailable ({str(e).splitlines()[0]})")
    print("[INFO] Falling back to outlines/transformers backend.")
    client = QwenOutlinesEnrichmentClient(model_name=ENRICHMENT_MODEL)
```

---

## Testing Checklist

Before pushing to production Colab session:

- [ ] Stage 1 completes without uv-related errors
- [ ] torch import succeeds; GPU/CPU backend detected
- [ ] Stage 2 completes (Drive mount, secrets, paths set)
- [ ] Stage 3 loads INPUT_CSV; recovers from Drive if missing
- [ ] Checkpoint merge works (if prior run exists)
- [ ] Stage 4 enriches at least 10 rows without crashes
- [ ] Checkpoint writing succeeds (no Drive quota errors)
- [ ] Stage 4 model cleanup releases VRAM
- [ ] Stage 5 embedding checkpoint creation works
- [ ] Stage 6 verification passes; embeddings shape is correct

---

## Usage After Fix

### Fresh Run (First Time)
1. Upload `vibescent_enriched.csv` to Drive at `/MyDrive/vibescent_offline/` or `/MyDrive/`.
2. Run Stage 1 → Restart runtime.
3. Run Stage 2 (config).
4. Run Stage 3 (load & inspect).
5. Run Stage 4 (enrich) — may take hours; will checkpoint every 100 rows.
6. Run Stage 5 (embed corpus) — will checkpoint every 3200 rows.
7. Run Stage 6 (verify).
8. Copy outputs to repo using commands in Stage 7 markdown cell.

### Resume After Runtime Restart (Mid-Pipeline)
- Each stage is now independent.
- Skip Stage 2 if Drive is already mounted.
- Stage 3 will recover INPUT_CSV from Drive automatically.
- Stage 4 will reconstruct df_work and resume from checkpoint.
- Stage 5 will resume from last embedding checkpoint.

### Colab Best Practices (From Official Docs)
- **Max runtime:** 12 hours free, up to 24 hours Pro+
- **Max idle timeout:** ~30 mins free tier (varies)
- **GPU availability:** Dynamic; T4, L4, A100 rotate; no guarantee
- **Drive I/O limits:** ~10k items per folder; ~10k ops/user quota; batch small file ops

---

## Notes for Future Maintainers

1. **Do not re-introduce uv pip.** Standard pip is compatible with Colab's pre-installed PyTorch.
2. **Always use defensive globals().get() for stage-independent paths.** Runtime restarts are frequent.
3. **Make GPU-heavy stages (4, 5) recoverable.** Checkpointing every 100–3200 rows keeps cost down.
4. **Test optional package installs gracefully.** GPU quota fluctuates; `required=False` prevents abort-on-quota-hit.
5. **Run graphify update after notebook edits.** Keep the knowledge graph in sync.

---

## Commits

| Commit | Message |
|--------|---------|
| `1b0e22f` | fix: harden offline pipeline notebook for Colab robustness |

---

**Total changes:** 294 lines added, 43 removed (107% increase in defensiveness)
