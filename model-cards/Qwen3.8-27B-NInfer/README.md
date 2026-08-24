---
library_name: ginfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - incoai/Qwen3.8-27B-DFlash2
base_model_relation: quantized
tags:
  - ginfer
  - qwen3.8
  - dflash2
  - multimodal
  - conversational
  - cuda
  - rtx-5090
---

# Qwen3.8-27B DFlash2 for GInfer

This model card is the version-controlled source for
[neroued/Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer).

The repository contains two registered GInfer artifacts produced from
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) and its fixed
[DFlash2 companion](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2). Both use the native
[GInfer](https://github.com/Neroued/ninfer) `.ginfer` artifact format. They are not Transformers
checkpoints, Safetensors distributions, or GGUF files.

Both artifacts keep the same groupwise Text, Vision, MTP, optimized proposal-head, tokenizer,
chat-template, generation, and media-processor inventory. They add the exact five-layer,
81-tensor DFlash2 companion. The Q4 profile stores all 49 companion matrices as native Q4G64; the
W8 profile stores them as W8G32. No runtime repacking occurs.

## Artifacts

| Field | Q4 companion (default) | W8 companion |
|---|---|---|
| Filename | `qwen3_8_27b_dflash2_q4.ginfer` | `qwen3_8_27b_dflash2_w8.ginfer` |
| Size | 19,233,276,416 bytes (17.91 GiB) | 20,255,474,176 bytes (18.86 GiB) |
| SHA-256 | `3268315d0c90b7981cb5d86edcb60a427dcfbabb2185ffcb3f1a1090855b0771` | `40023b7a56c675b4d8ec95a03194a37cf8e00a274738549ddbb51d3c02632612` |
| Container version | 2 | 2 |
| GInfer model ID | `qwen3.8-27b` | `qwen3.8-27b` |
| GInfer weights ID | `groupwise-int-dflash2-q4` | `groupwise-int-dflash2-w8` |
| GInfer target key | `qwen3_8_27b` | `qwen3_8_27b` |
| Stored objects | 1,205 (1,199 tensors, 6 resources) | 1,205 (1,199 tensors, 6 resources) |
| DFlash2 matrix format | `Q4G64_F16S` | `W8G32_F16S` |

The Q4 companion reduces resident DFlash2 matrix storage by 1,022,197,760 bytes (about 0.95 GiB)
relative to W8. Under `--spec none` or MTP, all 81 companion tensors are validation-only, so the
two profiles have identical device residency.

Verify downloaded files with:

```bash
sha256sum --check SHA256SUMS
```

## Requirements

- a current GInfer source build that registers the two exact DFlash2 identities above;
- 64-bit Linux;
- NVIDIA GeForce RTX 5090 or NVIDIA RTX PRO 6000 Blackwell (`sm_120a`), NVIDIA GeForce RTX 4090
  (`sm_89`), or NVIDIA GeForce RTX 3090 (`sm_86`), subject to the memory available for the selected
  startup profile;
- CUDA Toolkit 13.1 or newer.

GInfer does not provide an install target or packaged binary. See the
[repository README](https://github.com/Neroued/ninfer#build) for source-build dependencies.

## Download and run

Download the default Q4 companion:

```bash
hf download neroued/Qwen3.8-27B-NInfer \
  qwen3_8_27b_dflash2_q4.ginfer \
  --local-dir models
```

The default `--spec auto` policy selects DFlash2 with four draft tokens and the full proposal head:

```bash
./build/apps/ginfer models/qwen3_8_27b_dflash2_q4.ginfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256
```

Use explicit `--spec none` for autoregressive decode:

```bash
./build/apps/ginfer models/qwen3_8_27b_dflash2_q4.ginfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256 \
  --spec none
```

The retained MTP route is selected explicitly:

```bash
./build/apps/ginfer models/qwen3_8_27b_dflash2_q4.ginfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

DFlash2 and Vision are mutually exclusive. Image/video startup must use
`--spec none --vision` or `--spec mtp --draft-tokens N --vision`; leaving `--spec` at its automatic
default while enabling Vision is rejected. For structured chat history and HTTP serving, see the
[GInfer documentation](https://github.com/Neroued/ninfer/tree/master/docs).

## Supported use

Both artifacts support:

- text generation in thinking and non-thinking modes;
- default DFlash2 speculative decoding with four learned draft tokens, or an explicit learned
  window from one to seven; exact request-local suffix copy may adaptively extend target
  verification through position fifteen after two confirmed copy rounds;
- explicit autoregressive decode and MTP speculative decoding with one to five draft positions;
- image, multi-image, video, and mixed multimodal messages under `--spec none` or MTP;
- BF16 and INT8 group-64 KV cache;
- compatible-prefix reuse and CUDA Graph decode on non-DFlash routes;
- startup-bounded small-scale concurrent serving with true batched decode;
- the GInfer CLI and OpenAI-/Anthropic-compatible serving.

## Limits

- GInfer accepts these files only under their exact registered identities.
- One Engine owns one CUDA device and a startup-fixed capacity of one to eight active requests.
- The speculative backend, proposal head, Vision allocation, and model residency cannot change
  after startup.
- DFlash2 and Vision cannot be enabled together.
- Context allocation is subject to GPU memory and the selected artifact, backend, KV-cache type,
  and concurrency.
- GInfer does not provide large-scale or preemptive continuous batching, priority/QoS scheduling,
  multi-GPU execution, CPU/GPU offload, or distributed serving.
- GInfer parses generated tool calls but does not execute them.

## Provenance

| Field | Value |
|---|---|
| Base source | `Qwen/Qwen3.8-27B` |
| Base revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Base download source | `modelscope.cn/models/Qwen/Qwen3.8-27B` |
| DFlash2 source | `incoai/Qwen3.8-27B-DFlash2` |
| DFlash2 revision | `dedf8df68adfb1afeaf7b7480c0a0243108177b4` |
| Conversion recipe | `qwen3_8_27b-v1` |
| Ranking input SHA-256 | `c692dc76388132c910547589b4fb4a0503fbd6ad50aaac6a509bbcb192a8afa5` |

The converter validates both source configurations, the exact BF16 source tensor sets, frontend
resources, complete ordered object plan, and numeric recipes before opening its output. The
primary Q4 metadata is published in
[`artifact-manifest.json`](https://huggingface.co/neroued/Qwen3.8-27B-NInfer/blob/main/artifact-manifest.json);
the W8 metadata is in
[`artifact-manifest-w8.json`](https://huggingface.co/neroued/Qwen3.8-27B-NInfer/blob/main/artifact-manifest-w8.json).
The exact storage contract is maintained in the
[Qwen3.8-27B artifact reference](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/qwen3.8-27b-artifact.md).

## License

This GInfer artifact distribution is licensed under the Apache License 2.0. The base
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) repository is also licensed under
Apache-2.0. Users remain responsible for the source repositories' terms and applicable laws.
