# Qwen3.8-27B CLI vs llama.cpp (MTP3 instruct)

> Historical GInfer profile: this comparison predates the registered Qwen3.8 DFlash2 companion
> artifacts and names the retired `qwen3.8-27b/groupwise-int` identity. The companion artifacts
> retain the same base Text/MTP inventory, but the commands below are provenance for the recorded
> runs rather than current artifact-download instructions. Current Q4/W8 DFlash2 production rows
> are in the [Qwen3.8 DFlash2 result](2026-08-20-qwen38-27b-dflash2-cli.md).

Paired **CLI** comparison (not `llama-bench`) of GInfer `groupwise-int` against ggml-org llama.cpp
Q4_K_M GGUF. Same user text (~999 prompt tokens), **1000** generated tokens, thinking off, MTP
window **3**, official Unsloth Qwen3.8 **instruct** sampling.

This is not a 5090 260k serve campaign and not a five-seed corpus. All four hosts used ggml-org
llama.cpp `d59d455fd` (not the Unsloth `iq1-narrow` fork).

## Protocol

| Knob | GInfer | llama.cpp (ggml-org) |
|---|---|---|
| Weights | `qwen3_8_27b.ginfer` `groupwise-int` | Four-host sweep: NAS `Qwen3.8-27B-Q4_K_M.gguf`. 5090 rerun: UD V3 + `-md` sidecar (see below). |
| KV | `--kv-dtype int8` (group-64) | `-ctk q8_0 -ctv q8_0` |
| Context | `--max-context 16384 --kv-capacity auto --prefill-chunk 1024` | `-c 16384 -ngl 99` |
| Speculative | `--spec mtp --draft-tokens 3 --lm-head-draft` | `--spec-type draft-mtp --spec-draft-n-max 3` |
| Thinking | `--no-thinking` | `--jinja --reasoning off --reasoning-budget 0` |
| Sampling | `--temperature 0.7 --top-p 0.80 --top-k 20 --min-p 0 --presence-penalty 1.5` | `--temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.0 --presence-penalty 1.5 --repeat-penalty 1.0` |
| Output | `--max-new 1000` | `-n 1000` |
| Mode | `--messages` JSON | `--conversation --single-turn --no-display-prompt` plus `-f` prompt file |

ggml-org llama.cpp has **no** `--mtp`. Unsloth guide flags (`--mtp 4`) are the `iq1-narrow` fork,
not this tree.

Prompt recipe (191 `itemNNNN` tokens plus the instruction) tokenizes to **999** prompt tokens on
both CLIs.

## Summary (GInfer / llama.cpp)

| Host | GPU | GInfer image | Prefill tok/s | Decode tok/s | MTP accept |
|---|---|---|---|---|---|
| 4090 WSL `DESKTOP-DG588BU` | RTX 4090 | sm_89 | 2048 / 1047 | 106.65 / 75.25 | 49.9% / 53.8% |
| S1 `AIS-2-8592-L01` | RTX PRO 6000 Blackwell Workstation | sm_120a | 3745 / 1467 | 161.56 / 100.12 | 49.1% / 51.1% |
| S2 `AIS-1-2950X-L02` | RTX 3090 | sm_86 | 955 / 864 | 65.52 / 49.49 | 52.1% / 52.3% |
| Local WSL `Ron-9950X3D2` | RTX 5090 | sm_120a | 3156 / 1344 | 160.44 / 109.13 | 49.1% / 52.6% |
| Local WSL (UD V3 + MTP sidecar) | RTX 5090 | sm_120a | 3156 / **2083** | 160.44 / **116.23** | 49.1% / **49.1%** |

## RTX 4090 (`sm_89`, `DESKTOP-DG588BU` WSL2)

Date: 2026-08-19. GInfer image `CMAKE_CUDA_ARCHITECTURES=89`. llama.cpp rebuilt with CUDA 13.3
`nvcc` (CUDA 12.8 `nvcc` ICE on `mmf-instance-ncols_15.cu`).

| Engine | Build | Prompt tok | Gen tok | Prefill tok/s | Decode tok/s | MTP accept |
|---|---|---:|---:|---:|---:|---|
| GInfer | `ginfer-4090` sm_89 | 999 | 1000 | **2048.07** | **106.65** | 49.92% (599/1200), 2.50 tok/round |
| llama.cpp | b10173 `e9fa0781f` (28 Jul) | 999 | 1000 | 1627.8 | 76.6 | not printed |
| llama.cpp | b10523 `d59d455fd` (19 Aug) | 999 | 1000 | 1047.00 | 75.25 | **53.84%** (617/1146), mean len 2.62 |

GInfer instruct MTP3 also printed `finish reason` `output-limit`, overall throughput 101.47 tok/s,
sampling `temp=0.70 top_p=0.80 top_k=20 min_p=0.00 presence=1.50`. An earlier same-prompt GInfer
MTP3 CLI (not the instruct-sampling row) was 2058.88 / 112.69 tok/s at 54.53% accept (620/1137).

Verbose llama.cpp b10523 confirmed `draft-mtp` (`creating MTP draft context`,
`speculative.types=none,draft-mtp`) and empty think tags
(`generation_prompt` ends with `<think>\n\n</think>\n\n`). Slot timings:

```
prompt eval time =     954.15 ms /   999 tokens ( 1047.00 tokens per second)
eval time        =   13276.02 ms /  1000 tokens (   75.25 tokens per second)
draft acceptance = 0.53839 (617 accepted / 1146 generated), mean len = 2.62
```

### Commands

```bash
# GInfer
./build-89/apps/ginfer models/qwen3_8_27b.ginfer \
  --messages /tmp/cmp-p1000.json \
  --max-new 1000 --max-context 16384 --kv-dtype int8 --kv-capacity auto \
  --no-thinking --prefill-chunk 1024 \
  --temperature 0.7 --top-p 0.80 --top-k 20 --min-p 0 --presence-penalty 1.5 \
  --spec mtp --draft-tokens 3 --lm-head-draft

# llama.cpp ggml-org (not Unsloth iq1-narrow)
./build-cuda13/bin/llama-cli \
  -m /path/to/Qwen3.8-27B-Q4_K_M.gguf \
  -f /tmp/cmp-p1000.txt \
  -n 1000 -c 16384 -ngl 99 -ctk q8_0 -ctv q8_0 \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.0 \
  --presence-penalty 1.5 --repeat-penalty 1.0 \
  --jinja --reasoning off --reasoning-budget 0 \
  --conversation --single-turn --no-display-prompt
```

## S1 RTX PRO 6000 Blackwell (`sm_120a`, `AIS-2-8592-L01`)

Date: 2026-08-19. GInfer `build/apps/ginfer` (`120a`). llama.cpp was built with CUDA 13.3 for
`120a-real`.

| Engine | Prompt tok | Gen tok | Prefill tok/s | Decode tok/s | MTP accept |
|---|---:|---:|---:|---:|---|
| GInfer | 999 | 1000 | **3744.82** | **161.56** | 49.09% (595/1212), 2.47 tok/round; by pos 295,186,114 |
| llama.cpp b10523 `d59d455fd` | 999 | 1000 | 1467.13 | 100.12 | 51.10% (604/1182), mean 2.53; per pos 0.739, 0.477, 0.317 |

## S2 RTX 3090 (`sm_86`, `AIS-1-2950X-L02`)

Date: 2026-08-19. GInfer `build/apps/ginfer` (`86`). llama.cpp
`build-cuda13` CUDA 13.1, `86-real`.

| Engine | Prompt tok | Gen tok | Prefill tok/s | Decode tok/s | MTP accept |
|---|---:|---:|---:|---:|---|
| GInfer | 999 | 1000 | **954.50** | **65.52** | 52.14% (609/1168), 2.56 tok/round; by pos 293,195,121 |
| llama.cpp b10523 `d59d455fd` | 999 | 1000 | 864.06 | 49.49 | 52.32% (610/1166), mean 2.57; per pos 0.748, 0.517, 0.303 |

## Local RTX 5090 (`sm_120a`, `Ron-9950X3D2` WSL)

Date: 2026-08-19. Native GInfer `build-120a/apps/ginfer` (`CMAKE_CUDA_ARCHITECTURES=120a`).
llama.cpp rebuilt from ggml-org `d59d455fd` with CUDA 13.3 pip `nvcc`
(`CCCL_DISABLE_CTK_COMPATIBILITY_CHECK`; pip `cuda.h` reports CUDA_VERSION 13000 vs nvcc 13.3).

| Engine | Prompt tok | Gen tok | Prefill tok/s | Decode tok/s | MTP accept |
|---|---:|---:|---:|---:|---|
| GInfer | 999 | 1000 | **3155.53** | **160.44** | 49.09% (595/1212), 2.47 tok/round; by pos 295,186,114 |
| llama.cpp `d59d455fd` | 999 | 1000 | 1344.42 | 109.13 | 52.58% (611/1162), mean 2.57; per pos 0.747, 0.505, 0.322 |

The 109.13 tok/s row used `-m` on `Qwen3.8-27B-Q4_K_M.gguf` with `--spec-type draft-mtp` and **no** `-md`. That GGUF still has in-model MTP tensors, so draft-mtp created an MTP context against the target (`creating MTP draft context`). Unsloth UD Dynamic V3 splits MTP into a sidecar; `-m` on the V3 file alone will not load `mtp-*.gguf`.

### 5090 rerun: UD V3 + `-md` sidecar (2026-08-20)

Same protocol, llama.cpp `d59d455fd`, plus `-ngld 99` and `-lv 3`. Sidecar is a real Qwen3.5 MTP block (`blk.64.nextn.*`, own `token_embd` / `output.weight`, 18 tensors, 1.37 GiB).

```
loading draft model '.../Qwen3.8-27B-UD-Q4_K_M-V3-GGUF/mtp-Qwen3.8-27B-Q4_0.gguf'
prompt eval time =     479.58 ms /   999 tokens (  2083.09 tokens per second)
       eval time =    8594.83 ms /  1000 tokens (   116.23 tokens per second)
draft acceptance = 0.49092 (  595 accepted /  1212 generated), mean len =  2.47
```

| Engine | Weights | Prefill tok/s | Decode tok/s | MTP accept |
|---|---|---:|---:|---|
| GInfer (unchanged) | `qwen3_8_27b.ginfer` groupwise-int | **3155.53** | **160.44** | 49.09% (595/1212), 2.47 tok/round |
| llama.cpp, old Q4_K_M, in-model MTP | `Qwen3.8-27B-Q4_K_M.gguf` | 1344.42 | 109.13 | 52.58% (611/1162), mean 2.57 |
| llama.cpp, UD V3 + sidecar | `Qwen3.8-27B-UD-Q4_K_M.gguf` + `mtp-Qwen3.8-27B-Q4_0.gguf` | 2083.09 | 116.23 | **49.09% (595/1212), mean 2.47** |

UD V3 + sidecar is real MTP (log line `loading draft model`), ~7 tok/s above the old GGUF, and the accept counts match GInfer exactly. GInfer decode is still ~38% ahead (160 vs 116).

```bash
./build-cuda13/bin/llama-cli \
  -m /path/to/Qwen3.8-27B-UD-Q4_K_M.gguf \
  -md /path/to/mtp-Qwen3.8-27B-Q4_0.gguf \
  -ngl 99 -ngld 99 \
  -f /tmp/cmp-p1000.txt -n 1000 -c 16384 -ctk q8_0 -ctv q8_0 \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.0 \
  --presence-penalty 1.5 --repeat-penalty 1.0 \
  --jinja --reasoning off --reasoning-budget 0 \
  --conversation --single-turn --no-display-prompt --no-mmap \
  -lv 3
```
