---
library_name: ginfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - unsloth/Qwen3.8-27B-NVFP4
  - incoai/Qwen3.8-27B-DFlash2
base_model_relation: quantized
tags:
  - ginfer
  - qwen3.8
  - nvfp4
  - fp8
  - dflash2
  - w4a4
  - blackwell
  - multimodal
  - conversational
  - cuda
  - rtx-5090
model-index:
  - name: Qwen3.8-27B-nvfp4-NInfer
    results:
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: IFBench
          type: ifbench
        metrics:
          - type: accuracy
            value: 77.00
            name: Prompt-level strict (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: GInfer EvalScope 1.9.0
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: AIME 2025
          type: aime25
        metrics:
          - type: accuracy
            value: 96.67
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: GInfer EvalScope 1.9.0
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: AIME 2026
          type: aime26
        metrics:
          - type: accuracy
            value: 96.67
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: GInfer EvalScope 1.9.0
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: GPQA-Diamond
          type: gpqa_diamond
        metrics:
          - type: accuracy
            value: 90.40
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: GInfer EvalScope 1.9.0
      - task:
          type: image-text-to-text
          name: Image Text to Text
        dataset:
          name: ERQA
          type: erqa
        metrics:
          - type: accuracy
            value: 66.25
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: GInfer EvalScope 1.9.0
      - task:
          type: image-text-to-text
          name: Image Text to Text
        dataset:
          name: RealWorldQA
          type: real_world_qa
        metrics:
          - type: accuracy
            value: 83.53
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: GInfer EvalScope 1.9.0
---

# Qwen3.8-27B NVFP4 for GInfer

This model card is the version-controlled source for
[neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer).

This profile family contains the registered NVFP4 weight profile of
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). It combines the official BF16 checkpoint
with the fixed packed Text weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) in the native
[GInfer](https://github.com/Neroued/ninfer) `.ginfer` artifact format. The artifact is intended
only for GInfer; it is not a Transformers checkpoint, Safetensors distribution, or GGUF file. Two
additional registered artifacts preserve that complete mixed NVFP4/FP8 image exactly and add the
fixed five-layer [DFlash2 companion](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) in native
Q4 or Q8 storage.

These are weight profiles for the existing `qwen3_8_27b` target, not separate model targets. The
version-2 artifact identity selects the NVFP4 binder and execution leaves. The `nvfp4` prefix names
the complete registered backbone rather than claiming that every matrix has one format:
Text layers 0–55 use NVFP4 MLP weights, while the token embedding, attention input/output
projections, GDN Q/K/V/Z and output projections, full output head, and Text layers 56–63 MLP weights
use row-scaled FP8. BF16 control weights and the registered MTP and Vision allocations are retained.

## Artifacts

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b_nvfp4.ginfer` |
| Size | 21,492,695,040 bytes (20.02 GiB) |
| SHA-256 | `bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32` |
| Container version | 2 |
| GInfer model ID | `qwen3.8-27b` |
| GInfer weights ID | `nvfp4` |
| GInfer target key | `qwen3_8_27b` |
| Stored objects | 1,124 (1,118 tensors and 6 resources) |
| NVFP4 tensors | 112 |
| Row-scaled FP8 tensors | 146 |

The file contains the registered Text, Vision, MTP, optimized proposal-head, tokenizer,
chat-template, generation, and media-processor objects required by GInfer. Source-derived NVFP4 and
FP8 words are preserved without decode and requantization; only the official BF16 token embedding
is encoded locally as row-scaled FP8.

The DFlash2 profiles add exactly 81 companion tensors without changing any base object or payload
word:

| Field | Q4 companion (production default) | Q8 companion |
|---|---|---|
| Filename | `qwen3_8_27b_nvfp4_dflash2_q4.ginfer` | `qwen3_8_27b_nvfp4_dflash2_q8.ginfer` |
| Size | 22,515,440,128 bytes (20.97 GiB) | 23,537,637,888 bytes (21.92 GiB) |
| SHA-256 | `d4054ac1b8132cd217a3487a6761f07479fe8df9f254068b3ddba3cc85c9521d` | `fd73db65f6abd764c1430cfec0e89cbe7c705209e86f552566674b912a5c5f83` |
| GInfer weights ID | `nvfp4-dflash2-q4` | `nvfp4-dflash2-w8` |
| Stored objects | 1,205 (1,199 tensors and 6 resources) | 1,205 (1,199 tensors and 6 resources) |
| DFlash2 matrix format | `Q4G64_F16S` | `W8G32_F16S` |
| Resident DFlash2 startup | 21,398,354,432 bytes | 22,420,552,192 bytes |

`W8G32_F16S` is GInfer's exact Q8 companion format. Q4 saves 1,022,197,760 resident bytes relative
to Q8. With DFlash2 disabled, both profiles use the same 20,375,621,632-byte NVFP4 base residency.

Verify the artifacts with:

```bash
printf '%s  %s\n' \
  'bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32' \
  'qwen3_8_27b_nvfp4.ginfer' | sha256sum --check
printf '%s  %s\n' \
  'd4054ac1b8132cd217a3487a6761f07479fe8df9f254068b3ddba3cc85c9521d' \
  'qwen3_8_27b_nvfp4_dflash2_q4.ginfer' | sha256sum --check
printf '%s  %s\n' \
  'fd73db65f6abd764c1430cfec0e89cbe7c705209e86f552566674b912a5c5f83' \
  'qwen3_8_27b_nvfp4_dflash2_q8.ginfer' | sha256sum --check
```

## Requirements

- [GInfer](https://github.com/Neroued/ninfer) revision
  [`5d2c1f5`](https://github.com/Neroued/ninfer/commit/5d2c1f5590b8f4c3d106a75f65210eb4efb8f4e1)
  or later, built from source;
- 64-bit Linux;
- NVIDIA GeForce RTX 5090 or NVIDIA RTX PRO 6000 Blackwell (`sm_120a`);
- CUDA Toolkit 13.1 or newer.

NVFP4 is not supported on RTX 4090 (`sm_89`) or RTX 3090 (`sm_86`). Engine load throws
`nvfp4 is not supported on this GInfer image`.

GInfer does not provide an install target or packaged binary. See the
[repository README](https://github.com/Neroued/ninfer#build) for source-build dependencies.

## Download and run

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ginfer \
  --local-dir models

./build/apps/ginfer models/qwen3_8_27b_nvfp4.ginfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Companion profiles are distributed only as completed project-release artifacts. Run a published
companion artifact from the GInfer checkout:

```bash
./build/apps/ginfer models/qwen3_8_27b_nvfp4_dflash2_q4.ginfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256
```

Automatic startup selects DFlash2 with four draft tokens for either companion profile. Use
`--spec none` for autoregressive decode or explicit `--spec mtp --draft-tokens 3 --lm-head-draft`
for the retained MTP route.

For images, videos, structured chat history, and HTTP serving, see the
[GInfer documentation](https://github.com/Gadflyii/ginfer/tree/main/docs).

## Supported use

The three artifacts support:

- text generation in thinking and non-thinking modes;
- image, multi-image, video, and mixed multimodal messages;
- MTP speculative decoding with draft windows from one to five;
- BF16 and INT8 group-64 KV cache;
- CUDA Graph decode and compatible-prefix reuse;
- startup-bounded small-scale concurrent serving with true batched decode;
- the GInfer CLI;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages serving.

The Q4 and Q8 companions additionally support DFlash2 with learned windows from one to seven and
default to four. DFlash2 is text-only and mutually exclusive with Vision; select `--spec none
--vision` or MTP with `--vision` for image/video startup.

## Performance

On an RTX 5090 with CUDA 13.3, the matched 999-token prompt / 1,000-token production-sampling
request used INT8 KV, DFlash2 `k=4`, temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5,
and seed 42. Two Q4 runs averaged **7,715.83 prefill tok/s**, **154.43 decode tok/s**, and
**151.55 overall tok/s**. Two Q8 runs averaged **7,676.49**, **142.02**, and **139.58 tok/s**.
Each profile reproduced its output and acceptance trajectory exactly across both runs. These are
complete sampled-route measurements, not an isolated quantization-quality or kernel-format ratio;
the Q4 and Q8 trajectories differ. The scoped command and counters are in the
[Qwen3.8 DFlash2 result](../../docs/results/2026-08-20-qwen38-27b-dflash2-cli.md).

The MTP0 measurements below were collected at GInfer revision
[`f08597d`](https://github.com/Neroued/ninfer/commit/f08597d6eaafce5b875934aaa85854fcd5426df8),
and the MTP3 measurements at revision
[`32c9881`](https://github.com/Neroued/ninfer/commit/32c9881b6783949df4999422a764b3dcaa111b13).
Both campaigns used one NVIDIA GeForce RTX 5090, CUDA 13.1 compile/runtime, CUDA driver API 13.3,
stochastic sampling, INT8 group-64 KV, CUDA Graphs, a 1,024-token prefill chunk, and prefix reuse
disabled. MTP0 used no speculative backend and a 262,144-token context limit; MTP3 used a
131,072-token per-request context limit and three draft tokens.

### Concurrent MTP=3 corpus makespan

The fixed corpus contains three long-reasoning and twelve cross-scenario fixtures with five seeds
each, for 75 requests. Every concurrency point starts a fresh server and uses the same shuffle seed
and ordered HTTP send sequence. C=1 is the serial single-request corpus. Makespan includes prefill,
decode, workload transitions, and final drain.

| C | Requests | Decode tokens | Makespan | Requests/s | Decode tok/s | Avg batch | MTP acceptance | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 75 | 752,160 | 4,670.27 s | 0.0161 | 161.1 | 1.00 | 60.8% | 1.00× |
| 2 | 75 | 739,951 | 2,510.78 s | 0.0299 | 294.7 | 1.98 | 59.2% | 1.86× |
| 4 | 75 | 713,384 | 1,647.74 s | 0.0455 | 432.9 | 3.29 | 58.0% | 2.83× |
| 8 | 75 | 723,602 | 2,164.90 s | 0.0346 | 334.2 | 2.36 | 57.6% | 2.16× |

All 300 requests completed without a request, CUDA, or out-of-memory failure. C=4 gives the
shortest complete-corpus makespan. C=8 is limited by memory pressure, which makes its
complete-corpus result slower than C=4. Sampling is stochastic, so the fixed prompts and seeds do
not imply token-identical continuations across concurrency-specific numerical routes; exact
decode-token totals are shown above.

### Long-context baseline (MTP disabled)

Each value is the arithmetic mean ± sample standard deviation over five fixed seeds after server
warm-up.

| Prompt tokens | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|
| 7,680 | 8,340.4 ± 13.0 | 931.6 ± 1.6 | 71.2 ± 0.1 |
| 64,512 | 5,297.9 ± 259.2 | 12,281.1 ± 561.5 | 65.7 ± 0.8 |
| 130,048 | 3,544.7 ± 25.3 | 36,853.5 ± 259.4 | 59.6 ± 0.9 |
| 260,096 | 2,203.1 ± 13.4 | 118,354.8 ± 717.2 | 52.9 ± 2.3 |

### MTP=3 single-request long-reasoning decode

The C=1 point supplies five samples for each fixture. Values are arithmetic mean ± sample standard
deviation from server phase timings and speculative counters.

| AIME 2026 fixture | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Problem 1 | 1,465.4 ± 417.3 | 195.2 ± 4.6 | 76.0% ± 2.4% | 3.28 ± 0.07 |
| Problem 15 | 65,414.4 ± 271.9 | 151.4 ± 2.0 | 56.2% ± 1.1% | 2.69 ± 0.03 |
| Problem 30 | 50,023.4 ± 14,839.1 | 167.5 ± 23.7 | 64.6% ± 14.9% | 2.94 ± 0.45 |

### MTP=3 single-request cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|
| Code | 194.3 ± 6.1 | 76.4% ± 3.9% | 3.29 ± 0.12 |
| Story | 126.1 ± 10.9 | 37.4% ± 5.8% | 2.12 ± 0.17 |
| Translation | 192.3 ± 11.9 | 75.0% ± 6.5% | 3.25 ± 0.19 |
| Structured output | 219.8 ± 8.6 | 90.8% ± 5.1% | 3.72 ± 0.15 |

See the
[full methodology and results](https://github.com/Neroued/ninfer/blob/master/docs/performance.md)
for metric definitions and the exact reproduction command.

## Evaluation

The artifact was evaluated through GInfer's OpenAI-compatible serving route with thinking enabled,
MTP=3, and INT8 group-64 KV. EvalScope 1.9.0 used 0-shot prompts, rule-based scoring, and one sample
per problem with temperature 1.0, top-p 0.95, top-k 20, presence penalty 0.0, and seed 42. The text
suite ran at a 252,928-token context limit; the multimodal suite ran with `--vision` at a
81,920-token limit.

| Benchmark | GInfer NVFP4 | Correct / total | Official Qwen3.8-27B BF16 |
|---|---:|---:|---:|
| IFBench (prompt-level strict) | 77.00% | 231 / 300 | 79.5 |
| AIME 2025 | 96.67% | 29 / 30 | — |
| AIME 2026 | 96.67% | 29 / 30 | — |
| GPQA-Diamond | 90.40% | 179 / 198 | 89.2 |
| ERQA | 66.25% | 265 / 400 | 65.5 |
| RealWorldQA | 83.53% | 639 / 765 | 85.9 |

All 1,723 configured samples completed and were scored. IFBench additionally reports 80.50%
instruction-level strict, 80.33% prompt-level loose, and 83.50% instruction-level loose. These are
single-sample results, not pass@k.

The official Qwen3.8-27B BF16 figures come from the
[upstream model card](https://huggingface.co/Qwen/Qwen3.8-27B); its sampling settings and IFBench
metric level are not stated there, so the last column is not a same-protocol comparison. The NVFP4
deltas stay within ±2.5 points on the four overlapping benchmarks, and the upstream card reports no
AIME results.

## Limits

- The artifacts are accepted only by a GInfer revision that registers their exact identities and
  the matching target.
- GInfer executes on one RTX 5090 or RTX PRO 6000 Blackwell and one CUDA device, with a startup-fixed capacity of 1–8 active
  requests per Engine.
- It does not provide large-scale or preemptive continuous batching, priority/QoS scheduling,
  multi-GPU execution, CPU/GPU offload, or distributed serving.
- Context allocation is subject to GPU memory and the selected KV-cache type.
- GInfer does not execute generated tool calls.

## Provenance

| Field | Value |
|---|---|
| Base repository | `Qwen/Qwen3.8-27B` |
| Base revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Base download source | `modelscope.cn/models/Qwen/Qwen3.8-27B` |
| Quantized source repository | `unsloth/Qwen3.8-27B-NVFP4` |
| Quantized source revision | `60e813d4dbbdc5d64cf3f5a8caf2897bedf03679` |
| DFlash2 source repository | `incoai/Qwen3.8-27B-DFlash2` |
| DFlash2 source revision | `dedf8df68adfb1afeaf7b7480c0a0243108177b4` |
| Conversion recipe | `qwen3_8_27b_nvfp4-v1` |
| Companion composition recipe | `qwen3_8_27b_nvfp4_dflash2_compose-v1` |
| Embedding encoder | `MAXABS_BF16S_RECIP_E4M3FN_RNE_V1` |
| Converter repository | `https://github.com/Neroued/ninfer` |
| Base-artifact converter revision | `651d779657988dcb943896983d415ff6d38a21e2` |
| Base-artifact minimum runtime revision | `5d2c1f5590b8f4c3d106a75f65210eb4efb8f4e1` |
| Companion compositor and minimum runtime revision | `64212c6264cda5fa7a488502df210a1cc8778647` |
| Ranking input SHA-256 | `c692dc76388132c910547589b4fb4a0503fbd6ad50aaac6a509bbcb192a8afa5` |

The artifact identity, summarized object inventory, and conversion provenance are published in
[`artifact-manifest.json`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer/blob/main/artifact-manifest.json).
The exact storage contract is maintained in the
[Qwen3.8-27B artifact reference](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/qwen3.8-27b-artifact.md).

## License

This GInfer artifact is distributed under the Apache License 2.0. The
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) base repository and the
[quantized source repository](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) are also licensed
under Apache-2.0. Companion profiles additionally derive from
[incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2). Users remain
responsible for complying with the source licenses and applicable laws.
