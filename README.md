# GInfer

> Selected checkpoints. Maximum single-GPU inference performance.

GInfer is a fork of [Neroued/ninfer](https://github.com/Neroued/ninfer). It keeps the same binary
container (magic `NINFER`) but publishes tested product artifacts with the `.ginfer` extension. It
retargets the engine with extra SM families, occupancy from live SM count, and local product work
that is not going upstream.

GInfer is a from-scratch C++/CUDA inference engine for explicitly registered Qwen and Muse Glimmer
checkpoints on one GPU. The complete target set runs in the `sm_120a` image on NVIDIA GeForce RTX
5090. Separate NVIDIA GeForce RTX 4090 (`sm_89`) and RTX 3090 (`sm_86`) images cover groupwise
integer routes (no FP8 or NVFP4; 24 GB envelope). Qwen3.8 Q4-DFlash2 and Muse Q4-DFlash2 are also
physically qualified on RTX 3090; their `sm_89` images compile and admit the targets but still need
physical DFlash2 qualification. Qwen3.6-35B-A3B DFlash remains `sm_120a`-only. Qwen supports text,
image, and video prompts; Muse is text-only. Both use the local CLI and OpenAI-/Anthropic-compatible
HTTP APIs.

GInfer deliberately supports a closed set of model artifacts instead of acting as a general model
runtime:

| Model | Weights | GInfer artifact | Size | SHA-256 |
|---|---|---|---:|---|
| [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | `groupwise-int` | `qwen3_6_27b.ginfer` | 17,495,365,888 bytes (16.29 GiB) | `7b51600ffd10632b9660f56085efdd9b751d79733ad32036a652234b64bebe7b` |
| [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | `nvfp4` | `qwen3_6_27b_nvfp4.ginfer` | 18,324,064,000 bytes (17.07 GiB) | `bce5f00d066c0f20f1317bf1fdcb458264cf95837c3b1f3fbec163694627893a` |
| [Qwen3.8-27B DFlash2 Q4](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | `groupwise-int-dflash2-q4` | `qwen3_8_27b_dflash2_q4.ginfer` | 19,233,276,416 bytes (17.91 GiB) | `3268315d0c90b7981cb5d86edcb60a427dcfbabb2185ffcb3f1a1090855b0771` |
| [Qwen3.8-27B DFlash2 W8](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | `groupwise-int-dflash2-w8` | `qwen3_8_27b_dflash2_w8.ginfer` | 20,255,474,176 bytes (18.86 GiB) | `40023b7a56c675b4d8ec95a03194a37cf8e00a274738549ddbb51d3c02632612` |
| [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | `nvfp4` | `qwen3_8_27b_nvfp4.ginfer` | 21,492,695,040 bytes (20.02 GiB) | `bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32` |
| [Qwen3.8-27B NVFP4 + DFlash2 Q4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | `nvfp4-dflash2-q4` | `qwen3_8_27b_nvfp4_dflash2_q4.ginfer` | 22,515,440,128 bytes (20.97 GiB) | `d4054ac1b8132cd217a3487a6761f07479fe8df9f254068b3ddba3cc85c9521d` |
| [Qwen3.8-27B NVFP4 + DFlash2 Q8](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | `nvfp4-dflash2-w8` | `qwen3_8_27b_nvfp4_dflash2_q8.ginfer` | 23,537,637,888 bytes (21.92 GiB) | `fd73db65f6abd764c1430cfec0e89cbe7c705209e86f552566674b912a5c5f83` |
| [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | `groupwise-int` | `qwen3_6_35b_a3b.ginfer` | 22,783,246,080 bytes (21.22 GiB) | `1fb9ea0b5b8561e49d9604115ec89e5d9f2b6f6434e32c37c57fffd480a325d2` |
| [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) | `groupwise-int` | `muse_glimmer_30b.ginfer` | 18,047,450,368 bytes (16.81 GiB) | `93e0da71295039e759590cc00991dc8aae662e1081b6981344b76f1c82b4f06b` |
| [Muse Glimmer 30B low-VRAM DFlash](https://huggingface.co/meta-models/Muse-Glimmer-30B) | `groupwise-int-dflash-q4` | `muse_glimmer_30b_dflash_q4.ginfer` | 17,730,334,976 bytes (16.51 GiB) | `4f0c17d31326d9dc2e831eded373ba5802c3a4874d5b5844751de7caaf79f3bd` |
| [Muse Glimmer 30B NVFP4](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-NVFP4) | `nvfp4` | `muse_glimmer_30b_nvfp4.ginfer` | 18,516,895,232 bytes (17.25 GiB) | `c0971ae724c5505572adb4ed0dfcc537988b0e4cf86b361da286fd972424be0a` |

Qwen3.6-27B exposes two registered weight profiles; Qwen3.8-27B exposes four DFlash2 companion
profiles plus companion-free NVFP4. The version-2 artifact identity selects the profile without a
separate runtime flag; Qwen3.8 uses target key `qwen3_8_27b` while sharing the 27B execution
package. Both groupwise Qwen3.8 companion artifacts retain the existing groupwise Text, Vision,
MTP, and frontend inventory and add the exact 81-tensor five-layer DFlash2 companion. The Q4
identity stores all 49 companion matrices as native Q4G64; the W8 identity stores them as W8G32
and uses about 0.95 GiB more device memory when DFlash2 is resident. Qwen3.8 keeps the checkpoint's
learned DFlash2 window in `1..7` (four under `--spec auto`). Its lossless request-local context-copy
path can fill a target verification block through position 15 after two fully accepted copy
rounds, without making the five-layer drafter execute an unsupported wider block.

The two NVFP4 companion profiles preserve every object and payload word from the registered mixed
NVFP4/FP8 backbone and add the same exact Q4 or Q8 companion payload. The Qwen3.6 `nvfp4` profile
uses W4A4 Tensor Core MMA for prefill and A16 NVFP4 kernels for decode. The Qwen3.8 `nvfp4`
profiles preserve their source's mixed allocation: NVFP4 MLP weights in Text layers 0–55 and
row-scaled FP8 for the token embedding, attention input/output projections, GDN Q/K/V/Z and output
projections, output head, and remaining MLP weights. All seven 27B artifacts retain the same Text,
Vision, MTP, prefix-reuse, CLI, and serving routes; the four Qwen3.8 companion profiles additionally
implement DFlash2.

Muse Glimmer is a separate 131,072-token text target rather than a Qwen execution-package alias.
Its artifacts bundle the five-layer DFlash2 companion. The default `--spec auto` policy
materializes those 81 tensors for DFlash2 with four draft tokens; explicit `--spec none` validates
them without making them resident. Muse supports autoregressive decode, optional DFlash2, BF16/INT8 KV,
longest-common-token-prefix reuse, ATEM tools, and the same CLI/serving protocols. It rejects Vision
and MTP. The default `groupwise-int` identity keeps the DFlash convolution projections and selector
in BF16. The `groupwise-int-dflash-q4` identity stores those 13 matrices directly as native Q4G64,
saving 317,115,392 device bytes when DFlash is resident while leaving AR-only residency unchanged.
The `nvfp4` identity preserves all 416 RedHat transformer-block linear matrices as native NVFP4,
keeps the embedding and output head in W8, and uses the same Q4 DFlash2 companion as the low-VRAM
identity. It is `sm_120a`-only. The Muse artifacts are produced offline from their safetensors
sources and distributed separately; they are not published `neroued` downloads.

## Performance

The published measurements cover the three Qwen3.6 artifact profiles, Qwen3.8-27B NVFP4, and a
dedicated [Qwen3.8-27B DFlash2 CLI result](docs/results/2026-08-20-qwen38-27b-dflash2-cli.md).
On the matched RTX 5090 999/1,000 production-sampling request, Q4 DFlash2 reaches 3,114.89 prefill
and 134.85 decode tok/s; W8 reaches 3,115.91 and 138.26 tok/s. The Q4 route also completes the same
workload at 954.06 prefill and 53.82 decode tok/s with the production-default `k=4` on a physical
RTX 3090. An explicit learned `k=3` reaches 955.82 prefill and 60.30 decode tok/s on that one sampled
request; the default remains `k=4` rather than encoding a workload-specific result into policy.
With the NVFP4/FP8 backbone, two repeated matched runs average 7,715.83 prefill and 154.43 decode
tok/s for the Q4 companion, versus 7,676.49 and 142.02 tok/s for Q8. The Q4 combination is the
measured production choice on RTX 5090: it is 8.7% faster than Q8 decode, uses 0.95 GiB less
resident companion memory, and is 7.4% faster than the companion-free NVFP4 MTP3 control on this
complete sampled request.

Muse Glimmer has a separate [RTX 5090 AR/DFlash2 CLI
result](docs/results/2026-08-20-muse-glimmer-30b-cli.md). On its matched, single-seed 1,050/1,000
token workload, AR reaches 75.54 decode tok/s, BF16-DFlash reaches 90.74 tok/s, and native
Q4-DFlash reaches 99.12 tok/s. After the compact decode-kernel and DFlash2-selector retune,
five complete native-NVFP4 repetitions profiled at `k=4` reach 110.39--113.35 decode tok/s
(112.32 mean). Sampled trajectories
prevent treating those rows as isolated storage-format ratios or model-quality claims. Muse
groupwise identities are admitted by the public Engine route on `sm_86`,
`sm_89`, and `sm_120a`, so RTX 3090, RTX 4090, RTX 5090, and RTX PRO 6000 Blackwell hosts can
undergo real-device qualification and tuning. The Q4 DFlash2 identity is now physically
qualified on RTX 3090 at 534.21 prefill and 32.57 decode tok/s on the matched 1,050/1,000 request.
Native NVFP4 requires `sm_120a`; Muse BF16 DFlash on Ampere and all Muse `sm_89` routes remain
unqualified until their own real-device runs.

### Concurrent MTP3 decode

Saturated decode was measured on an RTX 5090 with INT8 group-64 KV cache, CUDA Graphs, MTP3, and
one 8,192-token generation per active request. The values below are aggregate committed decode
throughput from complete one-second intervals in which the actual decode batch remained equal to
the configured concurrency. MTP acceptance is aggregated over the complete request wave. Each
concurrency cell reports `decode tok/s / MTP acceptance`; profiles should be read independently.

| Model profile | C=1 tok/s / accept | C=2 tok/s / accept | C=4 tok/s / accept | C=8 tok/s / accept | C8 / C1 |
|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B `groupwise-int` | 185.8 / 68.2% | 247.0 / 69.0% | 309.5 / 68.4% | 535.0 / 68.3% | 2.88× |
| Qwen3.6-27B `nvfp4` | 202.4 / 69.3% | 399.7 / 71.4% | 699.7 / 69.3% | 1,146.9 / 68.6% | 5.67× |
| Qwen3.6-35B-A3B `groupwise-int` | 593.0 / 67.2% | 877.7 / 68.2% | 1,166.0 / 69.8% | 1,313.8 / 67.3% | 2.22× |
| Qwen3.8-27B `nvfp4` | 143.8 / 48.9% | 267.6 / 48.1% | 461.1 / 45.8% | 766.6 / 46.0% | 5.33× |

At C=8, Qwen3.6-35B-A3B reaches **1,313.8 aggregate decode tok/s**. Qwen3.6-27B NVFP4 reaches
**1,146.9 tok/s** and **5.67×** its C=1 throughput. Qwen3.8-27B NVFP4 has **45.8–48.9%** MTP
acceptance, versus **67.2–71.4%** across the other measured profiles, so aggregate committed
throughput reflects both execution performance and speculative acceptance.

### Single-request serving

The single-request corpus was measured on the same GPU with INT8 group-64 KV cache, CUDA Graphs,
and a 1,024-token prefill chunk. Each reported fixture uses five fixed seeds after server warm-up.
Targets and weight profiles are reported independently rather than as cross-target comparisons.
Requests were submitted serially to a persistent server. The Qwen3.8-27B NVFP4 MTP0 results use the
same dedicated serial corpus runner as the Qwen3.6 profiles; its MTP3 results come from the C=1 point
of the fixed concurrent-corpus campaign documented in [Performance](docs/performance.md).

**Qwen3.6-35B-A3B**

- MTP0 at a 7,680-token prompt: **15,544.3 prefill tok/s** and **271.1 decode tok/s**.
- MTP0 at a 260,096-token prompt: **5,157.1 prefill tok/s** and **188.2 decode tok/s**.
- MTP3 long reasoning: **620.3–726.2 decode tok/s** with **72.7–82.8% acceptance**.
- MTP3 structured output: **770.9 decode tok/s**, **89.1% acceptance**, and **3.67 tokens/round**.

**Qwen3.6-27B (`groupwise-int`)**

- MTP0 at a 7,680-token prompt: **3,218.1 prefill tok/s** and **77.6 decode tok/s**.
- MTP0 at a 260,096-token prompt: **1,614.8 prefill tok/s** and **54.8 decode tok/s**.
- MTP3 long reasoning: **161.9–175.4 decode tok/s** with **73.4–78.8% acceptance**.
- MTP3 structured output: **193.0 decode tok/s**, **88.7% acceptance**, and **3.66 tokens/round**.

**Qwen3.6-27B (`nvfp4`)**

- MTP0 at a 7,680-token prompt: **11,191.5 prefill tok/s** and **86.4 decode tok/s**.
- MTP0 at a 260,096-token prompt: **2,510.6 prefill tok/s** and **59.9 decode tok/s**.
- MTP3 long reasoning: **213.1–231.0 decode tok/s** with **76.3–81.1% acceptance**.
- MTP3 structured output: **252.2 decode tok/s**, **89.8% acceptance**, and **3.69 tokens/round**.
- Against groupwise-int on the same corpus and runtime options: **3.48× the 7,680-token prefill
  throughput**, **1.55× the 260,096-token prefill throughput**, and **30–32% higher MTP3 decode
  throughput**.

**Qwen3.8-27B (`nvfp4`)**

- MTP0 at a 7,680-token prompt: **8,340.4 prefill tok/s** and **71.2 decode tok/s**.
- MTP0 at a 260,096-token prompt: **2,203.1 prefill tok/s** and **52.9 decode tok/s**.
- MTP3 long reasoning: **151.4–195.2 decode tok/s** with **56.2–76.0% acceptance**.
- MTP3 structured output: **219.8 decode tok/s**, **90.8% acceptance**, and **3.72 tokens/round**.

See [Performance](docs/performance.md) for the full methodology, variability, reproduction command,
and per-fixture results.

## Evaluation

Capability scores were measured through GInfer's OpenAI-compatible serving route with thinking
enabled, MTP=3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem):

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

The Qwen3.8-27B DFlash2 companion profiles have not yet been added to this published evaluation
campaign.
The Qwen3.6 rows used temperature 0.6 and presence penalty 1.0; the Qwen3.8-27B NVFP4 row used
temperature 1.0 and presence penalty 0.0. The multimodal columns (ERQA and RealWorldQA) ran with
`--vision` at a 81,920-token context limit; the text columns used a 252,928-token limit for
Qwen3.8-27B NVFP4 and a 262,144-token limit for the Qwen3.6 rows.

These are single-sample results under that GInfer evaluation profile, not pass@k. See the model
cards and [full performance document](docs/performance.md) for correct/total counts and evaluation
notes.

## Requirements

GInfer currently requires:

- 64-bit Linux;
- NVIDIA GeForce RTX 5090 or NVIDIA RTX PRO 6000 Blackwell (`sm_120a`); qualified groupwise-int
  targets also support NVIDIA GeForce RTX 4090 (`sm_89`) and RTX 3090 (`sm_86`) within their
  24 GB envelopes (no FP8 or NVFP4). Qwen3.8 Q4-DFlash2 and Muse Q4-DFlash2 are qualified on
  RTX 3090. Their `sm_89` DFlash2 routes remain enabled for testing but unqualified;
- NVIDIA driver support for CUDA 13.1 and the CUDA Toolkit 13.1 or newer;
- CMake 3.28 or newer and a C++20-capable host compiler;
- `pkg-config`;
- FFmpeg development libraries: `libavformat >= 60`, `libavcodec >= 60`,
  `libavutil >= 58`, and `libswscale >= 7`;
- `libcurl >= 7.85`;
- Ninja, when using the commands below.

The default image is `CMAKE_CUDA_ARCHITECTURES=120a`. Ampere (`86`) and Ada (`89`) are extra
images, not a fatbin. The allowlist is `{86,89,120a}`. There is no install target or packaged
binary distribution; GInfer is run from its source build tree.

## Build

```bash
git clone https://github.com/Gadflyii/ginfer.git
cd ginfer

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

RTX 4090 and RTX 3090 qualified Qwen groupwise-int targets use extra `sm_89` / `sm_86` images:

```bash
cmake -S . -B build-89 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-89 --parallel

cmake -S . -B build-86 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-86 --parallel
```

The default configuration builds:

```text
build/apps/ginfer
build/apps/ginfer-serve
```

Tests, benchmarks, and maintainer tools are excluded from the default build.

## Docker

Build the runtime image on a 64-bit Linux host with an RTX 5090 or RTX PRO 6000 Blackwell (the
Docker default is `sm_120a`), a CUDA 13.1-compatible NVIDIA driver, Docker, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
# RTX 5090 / RTX PRO 6000 Blackwell (default)
docker build --tag ginfer:sm120a .

# RTX 4090, qualified Qwen groupwise-int targets only
docker build --build-arg GINFER_CUDA_ARCHITECTURE=89 --tag ginfer:sm89 .

# RTX 3090, qualified Qwen groupwise-int targets only
docker build --build-arg GINFER_CUDA_ARCHITECTURE=86 --tag ginfer:sm86 .
```

Each command produces one architecture-specific image, not a fat binary. Use the tag matching the
host GPU. The `sm_86` and `sm_89` builds admit Muse artifacts so their real-device AR, DFlash2, and
kernel qualification can run; those routes remain unqualified until that evidence
exists. Qwen3.6-35B-A3B still rejects DFlash on those images.

Download a model into `models/` as described below, then run the HTTP server:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ginfer:sm120a \
  ginfer-serve /models/qwen3_6_27b.ginfer \
  --host 0.0.0.0
```

Run the CLI from the same image:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --volume "$PWD/models:/models:ro" \
  ginfer:sm120a \
  ginfer /models/qwen3_6_27b.ginfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-new 256
```

## Download a model

Use the Hugging Face CLI to download one of the six published Qwen artifacts:

```bash
hf download neroued/Qwen3.6-27B-NInfer \
  qwen3_6_27b.ginfer \
  --local-dir models

# Or the 27B NVFP4 weight variant:
hf download neroued/Qwen3.6-27B-nvfp4-NInfer \
  qwen3_6_27b_nvfp4.ginfer \
  --local-dir models

# Or Qwen3.8-27B with the default Q4 DFlash2 companion:
hf download neroued/Qwen3.8-27B-NInfer \
  qwen3_8_27b_dflash2_q4.ginfer \
  --local-dir models

# Or the W8 DFlash2 companion variant:
hf download neroued/Qwen3.8-27B-NInfer \
  qwen3_8_27b_dflash2_w8.ginfer \
  --local-dir models

# Or Qwen3.8-27B NVFP4:
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ginfer \
  --local-dir models

# Or:
hf download neroued/Qwen3.6-35B-A3B-NInfer \
  qwen3_6_35b_a3b.ginfer \
  --local-dir models
```

The composed Qwen NVFP4+DFlash2 profiles and the three Muse profiles are distributed as completed
`.ginfer` artifacts by the project release process; GInfer does not contain checkpoint ingestion,
quantization, conversion, composition, or training code. If a registered profile is not published,
it is not available through this public repository.

Current GInfer builds accept only the version-2 artifact container. All Qwen downloads above are
version 2. Replace an older Qwen3.6 artifact by downloading the current file again from its Hugging
Face repository; the runtime does not migrate or repack weights.

Each `.ginfer` file contains the weights and frontend resources needed by GInfer. It is not a
Transformers checkpoint, Safetensors distribution, or GGUF file.

Each artifact is complete, while GPU residency is fixed at process startup. The default
`--spec auto` policy selects DFlash with four draft tokens for the four Qwen3.8 companion artifacts
and Muse Glimmer; it resolves to no speculative backend for the other registered profiles. Use
`--spec none` for autoregressive Qwen3.8 or Muse startup and to leave the bundled DFlash2 tensors
nonresident. Qwen3.8 can instead select MTP explicitly with `--spec mtp --draft-tokens N`.
Vision is disabled by default, so its weights, Vision scratch phase, and frozen request-transient
allocation are omitted. A Qwen3.8 companion artifact must use `--spec none --vision` or
`--spec mtp --draft-tokens N --vision`; its auto-selected DFlash2 backend is mutually exclusive
with Vision. Disabled capabilities cannot be enabled by a later request. Qwen3.6-35B-A3B provides
text-only DFlash in the `sm_120a` image; Muse provides text-only DFlash2 and rejects both Vision and
MTP.

## Run the CLI

```bash
./build/apps/ginfer models/qwen3_6_27b.ginfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Use `--messages FILE` instead of `--prompt` for chat history and, on Qwen targets, images or videos:

```bash
./build/apps/ginfer models/qwen3_6_27b.ginfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --vision
```

Muse Glimmer defaults to DFlash2 with four draft tokens. Select a different explicit DFlash2
window with one to fifteen draft tokens:

```bash
./build/apps/ginfer models/muse_glimmer_30b.ginfer \
  --prompt "Explain speculative decoding in three sentences." \
  --max-context 16384 --max-new 256 \
  --spec dflash --draft-tokens 15
```

Muse rejects `--vision` and `--spec mtp`.

Answer content is written to stdout. Loading progress, reasoning, timing, throughput, memory, and
speculative-decoding statistics are written to stderr. See the [CLI guide](docs/cli.md) and
[committed examples](examples/cli/) for structured input and runtime options.

## Run the HTTP server

```bash
./build/apps/ginfer-serve models/qwen3_6_27b.ginfer \
  --max-context 16384 \
  --kv-capacity auto \
  --max-concurrency 2 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

The public model ID defaults to the artifact's `identity.model_id`; use `--model-id` only to
publish a deployment-specific alias.

Then send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

The server also implements OpenAI Responses Core (typed Items, semantic SSE, local continuation
state, and function calls) plus Anthropic Messages and token counting. Multimodal input is available
when the loaded Qwen target starts with `--vision`; Muse is text-only. See [HTTP
serving](docs/serving.md).

## Capabilities

All eight registered Qwen artifact identities support:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill and CUDA Graph decode;
- startup-bounded small-scale concurrent serving with true batched decode;
- MTP speculative decoding with draft windows from one to five;
- BF16 and INT8 group-64 KV cache;
- model- and thinking-mode-aware official sampling defaults, with explicit greedy, temperature,
  top-k, top-p, min-p, and presence/frequency-penalty overrides;
- compatible-prefix reuse;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming and
  usage accounting;
- prompt-rendered function tools and parsed tool calls.

The 35B-A3B target additionally supports text-only DFlash speculative decoding with draft windows
from one to fifteen in the `sm_120a` image.

The four Qwen3.8 DFlash2 companion identities support text-only DFlash2 with learned draft windows
from one to seven and select four draft tokens under `--spec auto`. Exact request-local suffix copy
can adaptively expand target verification through position fifteen after confirmed copy rounds;
CLI and serving metrics report the learned window, maximum verification window, and expanded-round
count separately.

The registered Muse Glimmer identities support text generation, chunked prefill, batched decode,
BF16/INT8 group-64 KV, compatible longest-common-token-prefix reuse, ATEM function calls, all three
serving protocols, native `instruct` direct-response mode, four reasoning strengths, and optional
text-only DFlash2 with draft windows from one to fifteen. It has no Vision or MTP route. Muse
autoregressive decode uses CUDA Graphs; its current DFlash2 round does not.

## Current limits

- Only the eleven `(model_id, weights_id)` artifact identities listed above are accepted product
  identities.
- Execution is specialized for one RTX 5090, RTX PRO 6000 Blackwell, RTX 4090, or RTX 3090 and one
  CUDA device. RTX 3090 qualification includes the Qwen3.8 and Muse native-Q4 DFlash2 routes;
  `sm_89` admits those routes for physical testing but has not yet qualified them. Native NVFP4 and
  Qwen3.6-35B-A3B DFlash remain `sm_120a`-only. Target packages reject an identity/feature profile
  outside an admitted image, while the documentation distinguishes admission from measured
  qualification.
- One Engine owns one resident model and supports a startup-fixed capacity of 1–8 active requests.
  Decode-ready requests are compacted at round boundaries and executed in one batched model
  traversal.
- GInfer does not provide large-scale or preemptive continuous batching, priority/QoS scheduling,
  multi-GPU execution, CPU/GPU offload, or distributed serving.
- `--max-context` is the logical ceiling of each sequence and is configurable up to the Qwen
  targets' native 262,144-token limit or Muse's 131,072-token limit. `--kv-capacity N` explicitly
  sizes the shared Main Text KV pool for all active and retained sequences, while
  `--kv-capacity auto` selects the largest usable capacity from the memory remaining after weights
  are loaded while preserving 1 GiB of sizing headroom. Omission defaults to one `--max-context`
  worth of pages. On RTX 4090 / RTX 3090 (24 GB), a Qwen3.8 DFlash2 companion artifact started
  with `--spec none` has the same base groupwise residency as the retired plain profile and leaves
  about 6.53 GiB / 7.19 GiB after weights; automatic DFlash2 residency uses additional memory.
  The no-speculative 24 GB product context is `--spec none --max-context 131072 --kv-dtype int8
  --kv-capacity auto` (verified on RTX 4090). A 262,144-token allocation does not fit. The resolved
  pool is fixed at startup and is not divided statically among request lanes.
- Tool calls are parsed and returned to the client; GInfer does not execute tools.
- The C++ headers are used by the in-tree applications and are not distributed as an installed SDK.

## Documentation

- [Contributing](CONTRIBUTING.md)
- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [CLI examples](examples/cli/)

## License

GInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. The Qwen3.8 DFlash2 companion artifacts also use
the fixed BF16 companion from
[incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2). The Muse artifact is derived from
[meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) and
[z-lab/Muse-Glimmer-30B-DFlash2](https://huggingface.co/z-lab/Muse-Glimmer-30B-DFlash2); consult
those source repositories for their licenses. Vendored dependencies retain their own license files
under `third_party/`.
