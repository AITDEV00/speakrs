# SPEAKRS 0.7 — Project Handoff Brief

> Context document for continuing work in `/home/jyao/DGE/speakrs`.
> Everything marked "verified" was confirmed against actual source code on 2026-10-18/19.
> Companion clones: `/home/jyao/DGE/rust-diar/` (diar-native + its speakrs fork), `/home/jyao/DGE/asr-repos/` (4 ASR servers), `/home/jyao/DGE/diart` (streaming), `/home/jyao/DGE/pyannote-audio` (official toolkit).

---

## 1. The mission (why we are in this repo)

Build a **self-hosted diarization/transcription gateway** behind **LiteLLM** exposing an OpenAI-compatible route:

```
POST /v1/audio/transcriptions

transcribe=true,  diarize=false  → transcription only
transcribe=true,  diarize=true   → transcription + diarization
transcribe=false, diarize=true   → diarization only   ← implemented by NOBODY; our extension
```

- **speakrs 0.7 is the chosen diarization engine** (native CUDA, fastest known community-1 implementation, has Python bindings → easiest to wrap).
- **diar-native is the ops/HTTP design reference only** (older engine, see §4).
- speakrs has **no HTTP server** (verified: no axum/hyper dep) — the gateway layer is ours to build.

## 2. Deployment target: NVIDIA H200 MIG

| MIG profile | VRAM | Compute |
|---|---|---|
| `1g.18gb` | 18 GB | 1/7 |
| `2g.35gb` | 35 GB | 2/7 |
| `3g.71gb` / `4g.71gb` | 71 GB | 3/7, 4/7 |
| `7g.141gb` | 141 GB | full |

- Plan: start on **`1g.18gb`** via Kubernetes extended resource `nvidia.com/mig-1g.18gb: 1`. Verify slice inside container with `nvidia-smi -L`.
- **No H200/H200-MIG benchmark exists** for speakrs or diar-native (searched thoroughly). Compatibility is established, performance is not.
- VRAM headroom: diar-native (ORT-based) measured peak **4,236 MB**; speakrs native CUDA observed ~2–4 GB. Both fit in 18 GB with large margin.

## 3. Repo state (this clone)

- Fork: **github.com/AITDEV00/speakrs** (from `avencera/speakrs`), all 8 branches copied and verified identical to upstream: `master`, `add-migraphx-feature`, `archive/preview-qualification-2026-09-10`, `bisect-benchmarks`, `cuda-backend`, `ffi`, `t3code/remove-cudnn-cuda-backend`, `wip-fp16`.
- `HEAD` = `af8013f` "Add Python bindings and release speakrs v0.7.0 (#70)" — **the v0.7.0 release commit** (2026-10-15).
- **Upstream never tags 0.6/0.7** (only `v0.4.1`, `v0.5.0` exist; GitHub shows v0.5.0 as "Latest release"). A local annotated tag `v0.7.0` was created at `af8013f` in this clone.

## 4. speakrs 0.7 architecture (verified in source)

### 4.1 The 0.6 CUDA rewrite — why this engine

CHANGELOG 0.6.0 (2026-10-13), 91 commits after the fork point upstream diverged from diar-native's base:

- **ONNX Runtime is OFF the CUDA path entirely.** Native backend loads the driver + cuBLAS + cuDNN 9 at runtime via `cudarc 0.19.10`. No CUDA toolkit needed to build. Same for CPU: native Rust PyanNet / WeSpeaker / filterbank.
- ORT `2.0.0-rc.13` still exists but **optional** (MIGraphX and `load-dynamic` external sessions only).
- Weights are **safetensors**, not ONNX: `segmentation-3.0.safetensors` + `wespeaker-multimask-tail.safetensors` (+ PLDA + embedding metadata). `ModelManager` downloads these for CUDA modes; export via `scripts/cuda/export_weights.py --runtime-assets <dir>`.
- Kernel inventory: TF32 tensor-core + Winograd ResNet trunk convolutions; FP16 hidden activations and residual stream (Turing/Ada/4090/A100 paths); tiled LSTM with resident recurrence (A100); FFT filterbank; native vector I/O; coalesced FP16 epilogue stores; overlap embedding batches with next decode; overlap of batch decode/clustering with GPU inference.
- **Per-layer kernel selection**: tune file → built-in recipe (measured on T4, A100, 4060 Ti, 4090, 5060 Ti) → device-class default. `SPEAKRS_CUDA_FORCE_LIBRARY=1` forces cuDNN/cuBLAS. There is a **`tune` command** to generate a recipe for untested GPUs — use this on the MIG slice.
- 0.7 additions (verified CHANGELOG): **Python bindings** (`pip install speakrs`, abi3 wheels; Linux x86_64 wheels ship CPU + CUDA), `speakrs diarize` CLI command, `MODEL_REPOSITORY`/`MODEL_REVISION`/`required_model_files` public API, `QueueReceiver::recv_timeout`, fix for PTX/cubin double-embedding (halved binary size).

### 4.2 H200 (SM90) facts — read before GPU work

`src/inference/cuda/tier.rs`:
- Tiers recognized: `Sm75` (Turing), `Sm80` (Ampere), `Sm90` (Hopper/H100/H200), `Sm120` (Blackwell). Requires cc ≥ 7.5.
- **`SHIPPED: [Self; 1] = [Self::Sm75]`** — only sm75 PTX ships as production tier. Feature wiring: `cuda-sm90 = ["_cuda", "dep:speakrs-ptx-sm75", "dep:speakrs-ptx-sm80"]` — **there is no `speakrs-ptx-sm90` crate**. On H200 the driver JIT-compiles sm75/sm80 PTX.
- So: H200 is an intended, compiled-in target — but NOT Hopper-specialized kernels. Treat H100 datapoints as the closest proxy.
- Cargo features for us: `cuda` (= all tiers) or `cuda-sm90`; convenience aliases exist (`cuda-a100`, `cuda-rtx40`, etc.).
- Hopper evidence: upstream README says speakrs' own kernels measured **1.4–2.0× faster than cuDNN/cuBLAS on RTX 4090, L4, A10, and H100**.

### 4.3 Benchmarks (verified in `benchmarks/README.md`, `benchmarks/voxconverse-dev.txt`)

| Run | DER | Time | RTF |
|---|---|---|---|
| speakrs `cuda`, RTX 4090, VoxConverse dev (20.3h) | 7.0% | 40.9s | **1786×** |
| speakrs `cuda-fast` (2s hop — different behavior, 7.4% DER) | 7.4% | 22.0s | 3325× |
| pyannote community-1 CUDA, same GPU | 7.2% | 2301.3s | 32× |
| 10-file subset incl. startup: T4 294×, A10 432×, L4 310×, **A100-40GB 1296×**, 4090 1453× | | | |

- Methodology warning: 1786× is aggregate dataset throughput (batch API), NOT single-file latency. No published warm single-file number → we must measure.

### 4.4 Python API (the 0.7 surface we wrap)

`crates/speakrs-python/` — `API.md`:
```python
import speakrs
pipeline = speakrs.Pipeline.from_pretrained("cuda")
result = pipeline.diarize_file("meeting.wav")
print(result.rttm())
# batch: pipeline.diarize_files([...])  (overlaps decode across up to 4 threads)
# async: diarize_samples_async / diarize_file_async / diarize_files_async (cancellation-safe)
```
Requirements: HF token + accepted community-1 gating (the models derive from `pyannote/speaker-diarization-community-1`).

## 5. diar-native — the comparison & ops reference (verified)

- `attevon-llc/diar-native` v0.3.3 (main @ `2da475c`), vendored engine = `attevon-llc/speakrs` fork @ `a5f3f39`, which declares **version 0.5.0** and `cuda = ["ort/cuda"]` (ONNX Runtime 1.24.2 + CUDA EP). It does NOT use speakrs 0.7. Diverged at upstream `b0756b1`; upstream moved 91 commits ahead; fork has 9 commits (+1,538/−280 lines, 24 files).
- Fork patch set (commit `5517abc`): folded seg graphs 179→40 nodes (bit-exact), multimask batching fix, fbank session pool (32.1s→12.9s ES2004a), vectorized VBx + threaded clustering (pdist 53×, VBx 8.3×), exclusive-overlap fix, shared ORT sessions, thread-safe RuntimeConfig. Latest fork commits: fbank pool reuse (2026-09-23), CUDA EP memory knobs (2026-10-07).
- **diar-native measured** (`validation/RESULTS.md`): 184× RT on A6000 (66.5-min Karpathy file, patched build), 159× ES2004a; AMI 13.101% DER @ 89× on 3080 Ti; VoxConverse dev-216 4.847% DER; **peak VRAM 4,236 MB**; bit-identical determinism across reruns.
- **Its HTTP design is the template worth stealing**: axum routes `POST /diarize`, `POST /embed_window`, `GET /healthz`, `GET /readyz`; request = JSON with shared-volume `wav_path`/`media_path`/`audio_path` (aliases), `file_id`, optional `gender` flag, optional per-request `device`; response = `DiarizeOutput` JSON: `segments`, `exclusive_segments`, `centroids` (gamma-weighted unnormalized per-speaker centroids, for kNN/voiceprint consumers), `num_speakers`, `rttm` (string), `speaker_gender` (optional). Also: admission control/permits, path redaction in logs, docker-compose base+gpu+prod overlays, CPU image that hard-fails on `DIAR_DEVICES=cuda` (no silent CPU degradation).
- Note: diar-native is a sidecar expecting a **shared volume** — for a gateway we'll want upload/bytes input instead (speakrs Python `diarize_samples`/async paths cover this).

## 6. Gateway API conventions (verified across 4 cloned ASR servers)

All four real, cloned at `/home/jyao/DGE/asr-repos/`, all expose `POST /v1/audio/transcriptions` with community-1:

| Impl | Diarization control | Format |
|---|---|---|
| **AWS WhisperX DLC** (docs verified) | `diarize` form field (default false) + `min_speakers`/`max_speakers` | `verbose_json`; model baked at `/opt/models/pyannote/speaker-diarization-community-1`; 1 req/container, queue→503, 100MiB cap |
| `DCC-BS/bentoml-faster-whisper` | `diarization: bool = true` config; rejects `json_diarized` without diarization | **`json_diarized`** (NOT interchangeable with diarized_json) |
| `collectiveai-team/coro` | server-configured backend (`pyannote` default, or `nemo`); per-request via `response_format` | **`diarized_json`**; OpenAI-exact SSE (`stream=true`); **explicitly rejects pyannote for its streaming pipeline** (batch-only clustering); OpenAI-SDK drop-in (ignores temperature, include[], etc.) |
| `eboltachev/OpenAI-ASR` | default `diarized_json`; `async=true` → 202 + `/v1/audio/transcriptions/jobs/{id}` + `/result`; `return_speaker_embeddings`; `merge_same_speaker_gap`; air-gapped model dirs | **`diarized_json`** |

- **Adopt**: AWS-style `diarize`/`transcribe` + `min_speakers`/`max_speakers` + `verbose_json`-ish response; OpenAI-ASR's job/poll pattern for long files; coro's error-mapping discipline.
- **Diarization-only mode exists nowhere** — only coro's *benchmark harness* has a `diarization_only` flag. Our extension.

## 7. Streaming/live tier (separate from speakrs — batch only)

- community-1 **cannot** stream (needs whole file for clustering; coro enforces this, pyannote FAQ confirms).
- Live options: **diart** (`juanmc2005/diart`, pyannote-based, self-hostable websocket server `diart.serve`, 0.5–5s tunable latency; forked to AITDEV00/diart with all 11 branches, cloned at `/home/jyao/DGE/diart`; single-client limitation in its WebSocketAudioSource), **NVIDIA Streaming Sortformer** (`nvidia/diar_streaming_sortformer_4spk-v2` — coro's live backend), **pyannoteAI Live-1** (hosted, sub-300ms, 16kHz mono 100ms chunks).

## 8. pyannote-audio reference facts (official repo, v4.0, at `/home/jyao/DGE/pyannote-audio`)

- `community-1` pipeline = segmentation + embedding + **VBx clustering**; output `DiarizeOutput` = `speaker_diarization` + `exclusive_speaker_diarization` (exclusive = one speaker/frame, for STT reconciliation).
- Self-hosted speed: 31s per audio-hour on H100. Offline/air-gapped supported (clone HF repo, `from_pretrained(path)`).
- pyannoteAI hosted API is **NOT OpenAI-compatible**: `https://api.pyannote.ai/v1/diarize`, Bearer key, upload→job→retrieve flow.

## 9. Next steps (in order)

1. **Smoke test** speakrs 0.7 CUDA on the H200 MIG `1g.18gb` slice (Python bindings, `diarize_file` on a sample, check DER sanity + `result.rttm()`).
2. **Run `tune`** on the MIG slice to generate a kernel recipe (MIG slices expose reduced SM counts — tuned recipes matter more, not less).
3. **Benchmark protocol**: same slice, same recordings (e.g. VoxConverse dev subset + one long file), warm single-file latency, aggregate throughput, peak GPU memory, DER vs official community-1 baseline. Compare `cuda` mode (not just `cuda-fast` — 2s hop changes behavior).
4. **Build the gateway**: thin FastAPI (Python bindings) or axum wrapper exposing `/v1/audio/transcriptions` with the three modes; response modeled on diar-native's `DiarizeOutput` (`segments`, `exclusive_segments`, `centroids`, `num_speakers`, `rttm`) mapped into OpenAI `verbose_json` + `diarized_json`.
5. Keep diar-native open for: admission control, VRAM knobs, health/ready split, log redaction patterns.

## 10. Quick file map (speakrs 0.7)

- `src/inference/cuda/tier.rs` — PTX tier selection, `SHIPPED`, `SPEAKRS_CUDA_PTX_TIER` env override
- `src/inference/cuda/` — `tuning/`, `kernels.rs`, `segmentation/`, `embedding/`, `fbank`, `dnn`, `blas.rs`, `gemm.rs`
- `crates/speakrs-python/` — `API.md` (full binding surface), abi3 wheels
- `crates/speakrs-ptx-sm75|sm80|sm120` — shipped PTX tiers (no sm90 crate)
- `scripts/cuda/export_weights.py` — safetensors asset export (`--runtime-assets`)
- `benchmarks/` — methodology + full result tables; `justfile` for common commands
- `Cargo.toml` features: `cuda`, `cuda-sm90`, `cuda-rtx40`, `migraphx`, `load-dynamic`, `online`, `cpu`, `coreml`
