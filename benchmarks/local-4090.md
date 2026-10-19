# Local RTX 4090 benchmarks — speakrs 0.6.0 native CUDA

Date: 2026-10-19. Engine: speakrs `0.6.0` (commit `472617c`, = published crates.io 0.6.0),
branch `jya0-v0.6.0`. Hardware: NVIDIA GeForce RTX 4090 (sm_89, 49,140 MiB), driver 616.92,
no CUDA toolkit (driver-loaded cuDNN/cuBLAS, JIT-compiled sm80 PTX per `Recipe(Rtx4090)`).
Models: `avencera/speakrs-models` @ pinned revision (9 files, 31 MB) — public, no HF token needed.

Purpose: gateway serving-parameter selection for the H200 MIG deployment, ahead of the MIG
run itself. All tools used are committed examples on this branch:

| Example | What it measures |
|---|---|
| `examples/cuda_diarize.rs` | single-file cold/warm latency + RTFx |
| `examples/cuda_concurrent.rs` | N `clone_shared` handles (own CUDA streams) in flight |
| `examples/cuda_batch.rs` | `run_batch` overlapped pipeline over a file corpus |

Corpus: 15 AMI Mix-Headset WAVs (16 kHz mono 16-bit), 2.6–39.1 min each, plus one
concatenated 82.9-min file; ~490 min of audio total, kept in `test-audio/` (not committed).

## Single stream

| Mode | File | Duration | Cold | Warm | RTFx |
|---|---|---|---|---|---|
| cuda | fixtures/test.wav | 26 s | 65 ms | 24 ms | 1089× |
| cuda | ES2004a | 17.5 min | 868 ms | 846 ms | 1241× |
| cuda | IS1000b | 39.1 min | 1.958 s | 1.896 s | 1236× |
| cuda | combined-82min | 82.9 min | 4.382 s | 4.306 s | 1156× |
| cuda-fast | ES2004a | 17.5 min | 473 ms | 433 ms | 2423× |
| cuda-fast | IS1000b | 39.1 min | 1.010 s | 0.983 s | 2384× |
| cuda-fast | combined-82min | 82.9 min | 2.163 s | 2.088 s | 2383× |

- Pipeline load from warm HF cache: **0.35–0.5 s**. First-ever run (with 31 MB download +
  PTX JIT): 43.9 s.
- Scaling is linear in duration; only a ~7% RTFx dip at 83 min (clustering cost grows with
  speaker count/turns, not audio length).
- `cuda-fast` = **1.93–1.96× faster** than `cuda` on every file, matching the upstream
  benchmark delta (1786× vs 3325× batch RTFx on the 4090).

## Batch API (`run_batch`, decode/inference/clustering overlapped)

| Mode | Corpus | Audio | Cold batch | Warm batch | RTFx | Throughput |
|---|---|---|---|---|---|---|
| cuda | 12 files | 347.5 min | 16.391 s | 16.446 s | 1268× | 21.1 min/s |
| cuda-fast | 12 files | 347.5 min | 8.498 s | 8.486 s | 2457× | 41.0 min/s |
| cuda | 3 long files | 139.5 min | 7.025 s | 7.050 s | 1187× | 19.8 min/s |
| cuda-fast | 3 long files | 139.5 min | 3.624 s | 3.436 s | 2436× | 40.6 min/s |

## Concurrent workers (`clone_shared`, one CUDA stream per handle)

| Mode | Workers | Per-worker RTFx | Aggregate RTFx | Wall |
|---|---|---|---|---|
| cuda | 1 | 1184× | 1184× | 1.00 s |
| **cuda** | **2** | ~829× | **1658×** | 2.91 s |
| cuda | 3 | 439× | 1317× | 2.71 s |
| cuda | 4 | ~323× | 1293–1564× | 3.76 s |
| cuda | 8 | ~163× | 1308× | 10.87 s |
| **cuda-fast** | **2** | ~1262× | **2524×** | 1.10 s |
| cuda-fast | 4 | ~600× | 2401× | 2.17 s |
| cuda-fast | 6 | 411× | 2465× | 3.22 s |
| cuda-fast | 8 | 304× | 2429× | 4.29 s |
| cuda-fast | 12 | 202× | 2427× | 6.53 s |

(4-file workload: ES2002a 16.1 min, ES2003a 5.9 min, IS1003a 8.2 min, TS3005a 8.9 min.)

## Findings

1. **The GPU saturates around ~2500× regardless of path.** Batch overlap and 2 concurrent
   streams hit the same ceiling. Extra workers past saturation only stretch per-request
   latency (cuda at 8 workers: 163× per request) without adding aggregate throughput.
2. **`cuda-fast` scales flat under concurrency** (2400–2500× from 2 to 12 workers) because
   the 2 s hop leaves interleaving gaps — GPU utilization observed at 17% at 4 workers.
   `cuda` mode degrades faster; its 2-worker point (1658×) is its practical peak.
3. **VRAM is flat and small**: ~2.65–2.8 GB total across every test — constant vs duration,
   file count, and worker count (12 clones included). Compute, not memory, is the only
   constraint; the H200 `1g.18gb` MIG slice has ~15× memory headroom for this pipeline.
4. **Accuracy context** (upstream `benchmarks/README.md`): `cuda` = 7.0% DER,
   `cuda-fast` = 7.4% DER on VoxConverse dev. The 0.4-point trade buys ~1.95× speed.
5. **Speaker counts stayed consistent** across modes (ES2004a: 5/5; IS1000b: 4/4; the
   concatenated 82-min file merged two clusters in fast mode, expected for heterogeneous
   stitched audio). RTTM for the 26-s fixture is byte-identical to the documented expected
   output.

## Recommended gateway serving configuration

- **Default profile** — `cuda` (accuracy tier), worker pool of **2** (`clone_shared`),
  admission inflight cap 2: ~1658× aggregate, 2 concurrent requests at ~830× each.
- **Throughput profile** — `cuda-fast`, worker pool of **4**, inflight cap 4:
  ~2450× aggregate, 4 concurrent requests at ~600× each.
- Expose the mode per request (or per deployment) in the coro speakrs adapter.
- Per-request latency bound: keep inflight ≤ worker count; queue beyond the semaphore
  (coro's `AsrCapacityError` admission pattern fits directly).
- On the H200 MIG slice, expect a fraction of these numbers from the 1/7-compute
  `1g.18gb` profile — re-run these three examples there and then `speakrs cuda tune`.
