# GInfer maintainer tools

`tools/` contains read-only artifact inspection, independent Python references, numerical parity
diagnostics, benchmark orchestration, and serving smoke checks.
These tools are not part of the public download-and-run path; normal users should start with the
[project README](../README.md).

Run commands from the repository root with a Python 3.11 environment containing the dependencies
for the selected tool.

## Task index

| Task | Location |
|---|---|
| Inspect artifact metadata and objects | [`artifact/inspect.py`](artifact/inspect.py) |
| Run the 27B Python reference | [`reference/qwen3_6_27b/`](reference/qwen3_6_27b/README.md) |
| Run the 35B-A3B Python reference | [`reference/qwen3_6_35b_a3b/`](reference/qwen3_6_35b_a3b/README.md) |
| Compare 27B artifact/source Vision activations | [`parity/qwen3_6_27b/`](parity/qwen3_6_27b/README.md) |
| Run benchmark matrices | [`bench/`](bench/README.md) |
| Exercise a resident HTTP server | [`smoke/serve_contract.py`](smoke/serve_contract.py) |
| Exercise thinking preservation through a managed server | [`smoke/serve_thinking_preservation.py`](smoke/serve_thinking_preservation.py) |

## Artifact boundary

Checkpoint ingestion, quantization, conversion, producer verification, composition, release
building, and v1-to-v2 migration are owned by a separate private producer repository. Its versioned
contract pins produced artifacts to this repository's framing, numeric formats, layouts,
registered inventories, and consumer qualification requirements.

Inspect a completed artifact locally without producer dependencies:

```bash
python3 -m tools.artifact.inspect out/qwen3_6_27b.ginfer --objects
```

Exact consumer inventories and formats remain recorded in [`docs/maintainer/`](../docs/maintainer/).
Published users normally download completed artifacts instead of running producer workflows.

## Python references and parity

```bash
python3 -m tools.reference.qwen3_6_27b \
  --weights out/qwen3_6_27b.ginfer \
  --prompt "请简短介绍一下你自己。" --decode 128

python3 -m tools.reference.qwen3_6_35b_a3b \
  --weights out/qwen3_6_35b_a3b.ginfer \
  --prompt "请简短介绍一下你自己。" --decode 128
```

The Python implementations are independent diagnostic references, not alternate public inference
products or generated-token goldens for the C++ engine. See the parity README for the direct 27B
artifact/source Vision comparison command.

## Benchmark orchestration

`tools/bench/run_ginfer_bench_matrix.py` builds and runs the public-Engine benchmark matrix and
writes ignored local reports below `profiles/bench/`:

```bash
python3 tools/bench/run_ginfer_bench_matrix.py --preset core --dry-run
python3 tools/bench/run_ginfer_bench_matrix.py --preset core
```

See [`tools/bench/README.md`](bench/README.md) and [`bench/README.md`](../bench/README.md) for the
orchestrator and executable contracts.

## Serving smoke

After starting `ginfer-serve` in another terminal:

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 \
  --model qwen3.6-27b
```

The client exercises OpenAI, Anthropic, streaming, usage, multimodal, and tool-call response
surfaces against the resident process.

For typed rewrite-checkpoint and thinking-history behavior, the managed smoke script launches a
real server and consumes the repository fixture:

```bash
python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_27b.ginfer --backend mtp
```
