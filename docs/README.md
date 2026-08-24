# GInfer documentation

Start with the [project README](../README.md) to build GInfer, download a published artifact, and
run the CLI or HTTP server.

## User guides

| Document | Purpose |
|---|---|
| [CLI](cli.md) | target-specific text/media input, output streams, sampling, MTP/DFlash, and common runtime options |
| [HTTP serving](serving.md) | target-specific OpenAI Responses/Chat Completions and Anthropic Messages behavior, state, streaming, token counting, authentication, and tool calls |
| [Performance](performance.md) | RTX 5090 single-request and concurrent-decode results, RTX PRO 6000 Blackwell hardware facts, RTX 4090 / RTX 3090 groupwise-int 24 GB envelopes, MTP/DFlash measurements, and reproduction commands |
| [CLI vs llama.cpp (MTP3 instruct)](results/2026-08-19-qwen38-27b-cli-vs-llamacpp.md) | Paired GInfer vs ggml-org llama.cpp 1k/1k CLI on 4090, S1, S2, and 5090 |
| [Qwen3.8-27B DFlash2 CLI](results/2026-08-20-qwen38-27b-dflash2-cli.md) | RTX 5090 groupwise and NVFP4-backbone Q4/Q8 DFlash2, AR, and MTP3 999/1000 production rows; RTX 3090 groupwise Q4 qualification and llama.cpp MTP3 comparison |
| [Muse Glimmer 30B CLI (AR and three DFlash2 storage profiles)](results/2026-08-20-muse-glimmer-30b-cli.md) | RTX 5090, 1050/1000 INT8: AR 75.54, BF16-DFlash 90.74, native Q4-DFlash 99.12, and RedHat NVFP4/Q4-DFlash 112.98 decode tok/s; RTX 3090 Q4 and 128K evidence included |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode, and long-context inputs |

The executable `--help` output is the exact source for command-line option spelling and defaults.

## Model artifacts

| Model | Weights | Download | Versioned model card source |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | [model card](../model-cards/Qwen3.6-27B-NInfer/README.md) |
| Qwen3.6-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) |
| Qwen3.8-27B | `groupwise-int-dflash2-q4`, `groupwise-int-dflash2-w8` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | [model card](../model-cards/Qwen3.8-27B-NInfer/README.md) |
| Qwen3.8-27B | `nvfp4`, `nvfp4-dflash2-q4`, `nvfp4-dflash2-w8` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) |
| Qwen3.6-35B-A3B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | [model card](../model-cards/Qwen3.6-35B-A3B-NInfer/README.md) |
| Muse Glimmer 30B | `groupwise-int`, `groupwise-int-dflash-q4`, `nvfp4` | distributed separately by the project release process | this release |

## Repository-local guides

- [Benchmarks](../bench/README.md)
- [Tests](../tests/README.md)
- [Maintainer tools](../tools/README.md)
- [Capability evaluation](../eval/README.md)
