# GInfer CLI

`build/apps/ginfer` runs one request against one registered `.ginfer` artifact. Build GInfer and
download an artifact using the [project README](../README.md) before following this guide.

## Text input

```bash
./build/apps/ginfer models/qwen3_6_27b.ginfer \
  --prompt "Summarize the difference between prefill and decode." \
  --max-context 16384 \
  --max-new 256
```

Exactly one of `--prompt` and `--messages` is required.

Answer content is streamed to stdout. Reasoning, model loading (including the registered target and
canonical `weights_id`), timings, throughput, GPU memory, and speculative-decoding statistics are
written to stderr, so stdout can be redirected independently:

```bash
./build/apps/ginfer models/qwen3_6_27b.ginfer \
  --prompt "Return one sentence." --max-new 64 \
  > answer.txt 2> run.log
```

Thinking is enabled by default. If the chat template embedded in the loaded artifact exposes
reasoning effort, `--reasoning-effort low|medium|high|xhigh` selects it; omitting the option uses the
template's default. An artifact whose template does not expose effort rejects the option. On a
frontend with a thinking toggle, add `--no-thinking` for direct-response prompt rendering; it
cannot be combined with `--reasoning-effort`. Muse exposes reasoning strengths with `high` as its
default; `--no-thinking` selects its native `instruct` strength. `--greedy` selects exact argmax
decoding independently.

## Startup memory profile

GPU residency is frozen when the Engine starts:

- omitting `--spec` is `--spec auto`: the four Qwen3.8 DFlash2 companion profiles and Muse Glimmer
  select DFlash with four draft tokens, while every other registered profile selects no
  speculative backend;
- `--spec none` explicitly selects autoregressive decode and leaves MTP/DFlash weights and state
  nonresident;
- `--spec mtp` loads only MTP (Qwen targets that implement it);
- `--spec dflash` loads the selected target's DFlash backend: Qwen3.8 or Muse Glimmer DFlash2, or
  35B-A3B text-only DFlash;
- a speculative backend with the full proposal head omits the optimized proposal head;
- Qwen Vision is disabled by default, omitting its weights, Vision scratch phase, and frozen
  request-transient allocation;
- `--vision` loads those allocations and enables image/video input on Qwen; Muse rejects it.

The complete `.ginfer` inventory is still validated. A Qwen3.8 companion artifact validates all 81
DFlash2 tensors even under `--spec none` or `--spec mtp`, but only DFlash startup materializes
them. The `groupwise-int-dflash2-q4` and `groupwise-int-dflash2-w8` profiles share the groupwise
Text, MTP, and Vision inventory. The `nvfp4-dflash2-q4` and `nvfp4-dflash2-w8` profiles instead
preserve the registered mixed NVFP4/FP8 backbone exactly. In either backbone family, Q4G64 stores
the 49 companion matrices in 1,022,197,760 fewer bytes than Q8/W8G32. The NVFP4 base uses
20,375,621,632 resident bytes with DFlash disabled; Q4 DFlash uses 21,398,354,432 bytes and Q8 uses
22,420,552,192 bytes.

Muse AR startup likewise validates its 81 DFlash2 tensors but leaves them nonresident. The
`groupwise-int-dflash-q4` identity stores the ten convolution projections and three selector
matrices as native Q4G64, reducing DFlash residency by 317,115,392 bytes while leaving AR-only
residency unchanged. The `groupwise-int` identity retains those matrices in BF16. The `nvfp4`
identity preserves the RedHat transformer-block linears as native NVFP4 and pairs them with the Q4
DFlash matrices; it requires an `sm_120a` image. All three profiles are qualified on RTX 5090. The
measured differences depend on draft window and sampled acceptance trajectory, so use the [scoped
CLI result](results/2026-08-20-muse-glimmer-30b-cli.md) rather than treating any format as
universally faster. These choices are not lazy loading: a text-only Engine rejects media and cannot
enable Vision later. DFlash and Vision are mutually exclusive. For a Qwen3.8 companion artifact,
Vision startup must therefore use `--spec none --vision` or
`--spec mtp --draft-tokens N --vision`; leaving `--spec` at auto selects DFlash2 and is rejected
with Vision.

## Structured messages

`--messages` accepts either a non-empty JSON message array or an object containing `messages`
and an optional `tools` array.

```json
[
  {
    "role": "system",
    "content": "Answer concisely."
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image",
        "image": "examples/cli/media/visual_chart.png"
      },
      {
        "type": "text",
        "text": "Describe the chart."
      }
    ]
  }
]
```

Run message files from the repository root when they contain repository-relative media paths:

```bash
./build/apps/ginfer models/qwen3_6_27b.ginfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --vision
```

Supported roles are `system`, `developer`, `user`, `assistant`, and `tool`.
System and developer messages retain their array positions; the Qwen family frontend renders both
as system-class ChatML turns rather than moving later instructions to the beginning. Muse lowers
developer instructions to its system-class ATEM framing and accepts text parts only.

Qwen message content may be a string or an ordered array containing:

| Content type | Source field | Accepted source |
|---|---|---|
| text | `text` | string |
| image / image_url | `image` or `image_url` | local path, HTTP(S) URL, or base64 data URI |
| video / video_url | `video` or `video_url` | local path, HTTP(S) URL, or base64 data URI |

`image_url` and `video_url` may be strings or objects containing a string `url`. Assistant
history may include `reasoning_content` and `tool_calls`; a tool result uses role `tool` and
`tool_call_id`. Muse accepts the same text/history/tool structure but rejects image and video
parts.

See [`examples/cli/`](../examples/cli/) for committed text, image, video, mixed-media, thinking,
long-decode, and long-context inputs.

## Speculative decoding

The default `--spec auto` policy is target-specific. It selects DFlash with four draft tokens for
the four Qwen3.8 DFlash2 companion artifacts and Muse Glimmer, and no speculative backend for the other
registered profiles. Use `--spec none` to force autoregressive decode. Select MTP explicitly with
one to five draft positions. Explicit DFlash windows are one to seven for Qwen3.8 and one to
fifteen for Qwen3.6-35B-A3B or Muse. `--lm-head-draft` selects the Qwen optimized proposal head,
requires an explicit MTP/DFlash backend, and cannot be combined with `--spec auto`:

```bash
./build/apps/ginfer models/qwen3_6_35b_a3b.ginfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 \
  --max-new 512 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

For DFlash:

```bash
./build/apps/ginfer models/qwen3_6_35b_a3b.ginfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 --max-new 512 \
  --spec dflash --draft-tokens 7 --lm-head-draft
```

Qwen3.8 companion artifacts need no speculative flags for their default DFlash2 `k=4` route:

```bash
./build/apps/ginfer models/qwen3_8_27b_dflash2_q4.ginfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 --max-new 512
```

Use `qwen3_8_27b_nvfp4_dflash2_q4.ginfer` in the same command for the measured NVFP4/FP8 backbone
with the lower-memory Q4 companion.

For Qwen3.8, `--draft-tokens` controls only the learned five-layer drafter and therefore accepts
`1..7`. The runtime also performs exact request-local suffix lookup. After every row in the compact
batch has fully accepted two context-copy rounds, the target verifier can expand through position
15 while the learned drafter remains at its configured width. Batch-one expansion uses a dedicated
CUDA Graph; expanded batches of two through eight run eagerly. CLI summaries separately expose the
learned draft window, maximum verification window, and expanded-round count.

Use `--spec none` for Qwen3.8 autoregressive decode, or select its retained MTP route explicitly:

```bash
./build/apps/ginfer models/qwen3_8_27b_dflash2_q4.ginfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 --max-new 512 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

MTP and DFlash cannot be enabled together. Muse Glimmer 30B rejects `--spec mtp`; its DFlash2
window is `1..15`, with four selected by auto and by the measured production profile. That path
currently skips CUDA Graph capture. Qwen3.6-35B-A3B's DFlash route remains text-only and
`sm_120a`-qualified. The older Qwen3.6 [performance results](performance.md) use MTP with three
draft tokens and DFlash with seven draft tokens (block length eight), both with the optimized
proposal head. Current Qwen3.8 and Muse CLI rows are in
[the Qwen3.8 DFlash2 result](results/2026-08-20-qwen38-27b-dflash2-cli.md) and
[the Muse result](results/2026-08-20-muse-glimmer-30b-cli.md).

## Common options

| Option | Meaning | Default |
|---|---|---:|
| `--max-context N` | per-sequence logical context ceiling | `2048` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `2048` |
| `--prefill-chunk N` | positive text-prefill chunk, in multiples of 128 | `1024` |
| `--max-new N` | requested output-token limit | `128` |
| `--device N` | CUDA device index | `0` |
| `--kv-dtype bf16\|int8` | KV-cache storage | `bf16` |
| `--spec auto\|none\|mtp\|dflash` | speculative backend; auto selects Qwen3.8/Muse DFlash `k=4`, otherwise none | `auto` |
| `--draft-tokens N` | MTP `1..5`; Qwen3.8 DFlash `1..7`; other DFlash `1..15` | unset |
| `--lm-head-draft` | Qwen optimized proposal head | off |
| `--vision` | enable Qwen image/video input and load Vision GPU allocations | off |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--no-thinking` | disable thinking when the loaded frontend supports the toggle | thinking on |
| `--reasoning-effort low\|medium\|high\|xhigh` | select an effort exposed by the loaded chat template | template default |
| `--greedy` | exact argmax decoding | off |
| `--temperature F` | sampling temperature override | registered model/mode default |
| `--top-p F` | nucleus-threshold override | registered model/mode default |
| `--top-k N` | top-k-threshold override | registered model/mode default |
| `--min-p F` | min-p-threshold override | registered model/mode default |
| `--presence-penalty F` | presence-penalty override | registered model/mode default |
| `--frequency-penalty F` | frequency-penalty override | registered model/mode default (`0`) |
| `--seed N` | sampling seed | `0` |

When a sampling flag is omitted, Engine selects the official general-task preset registered for
the loaded model and the rendered prompt mode. The current presets are:

| Model | Prompt mode | Temperature | Top-p | Top-k | Min-p | Presence penalty |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.6-27B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.8-27B | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.8-27B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | thinking | `1.0` | `0.95` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Muse Glimmer 30B | ATEM reasoning | `1.0` | `0.95` | `64` | `0` | `0` |

Frequency penalty is `0` in every registered preset. Qwen's separate precise-coding recommendation
is task-specific and is therefore an explicit override rather than an inferred Engine default.

Repeat `--stop-token-id`, `--stop`, or `--reasoning-stop` to add stop conditions. Use
`--raw-output` to expose the frontend's raw output stream and `--print-token-ids` to include
generated token IDs in diagnostics.

Run `./build/apps/ginfer --help` for the exact option contract.

## Context and memory

The Qwen model IDs have a native context limit of 262,144 tokens; Muse Glimmer has a 131,072-token
limit. The practical allocation on one RTX 5090 depends on the selected artifact, media workload,
output budget, and KV-cache type. Muse's native-Q4 DFlash2 profile is qualified on `sm_120a` and
physical `sm_86`; its other Ampere profiles and all `sm_89` Muse routes remain admitted for tests
but unqualified. Qwen3.8 native-Q4 DFlash2 is likewise qualified on physical `sm_86`.
Qwen3.6-35B-A3B DFlash retains its `sm_120a` gate, while its other groupwise routes remain
available on their qualified images.

On RTX 4090 (24 GB, `sm_89` groupwise integer), a Qwen3.8 DFlash2 companion artifact under
`--spec none` uses the same 15.92 GiB base-weight residency as the retired plain profile and leaves
6.53 GiB (7,011,174,400 bytes) free after load. `--kv-capacity auto` still cannot exceed
`--max-context` (C=1). The default CLI `--max-context 2048` therefore admits **2048** tokens. With
1 GiB automatic headroom, no Vision, and `--spec none`, the largest Engine starts measured for
that base residency were **82,688** BF16 tokens and **160,448** INT8 group-64 tokens.
`--max-context 262144` does **not** start. The no-speculative 24 GB product context is
**`--spec none --max-context 131072 --kv-dtype int8 --kv-capacity auto`** (verified Engine start:
KV capacity 131,072, INT8 payload 4.12 GiB, 1.93 GiB free after startup). BF16 cannot admit 131,072
on this SKU. The default auto-selected DFlash2 route materializes the companion and has a smaller
KV envelope.

On RTX 3090 (24 GB, `sm_86` groupwise integer), the same Qwen3.8 companion under `--spec none`
uses 15.92 GiB of base weights and leaves 7.19 GiB free after load. `--kv-capacity auto` still
cannot exceed `--max-context` (C=1). The default CLI `--max-context 2048` therefore admits
**2048** tokens. With 1 GiB automatic headroom, no Vision, and `--spec none`, the largest Engine
starts measured for that base residency were **93,568** BF16 tokens and **181,504** INT8 group-64
tokens. `--max-context 262144` does **not** start. The no-speculative 24 GB product context is
**`--spec none --max-context 131072 --kv-dtype int8 --kv-capacity auto`** (INT8 start limit on this
SKU is 181,504). BF16 cannot admit 131,072 on 24 GB; auto-selected DFlash2 reduces this envelope.

Use `--kv-dtype int8` for large context allocations. The prepared prompt must fit
`--max-context`; generation stops at the remaining context capacity when necessary.
`--kv-capacity N` controls the shared physical Main Text KV pool independently and is rounded up to
the 64-token page size. `--kv-capacity auto` loads the selected weights, measures the remaining GPU
memory, and directly chooses the largest legal page capacity for the complete enabled runtime
layout. This includes the selected speculative backend, fixed sequence state, workspace, Vision
request transient, and CUDA Graph allowance, while leaving the default 1 GiB automatic headroom
unallocated. It does not probe allocations or resize the pool at request time. The single-request
CLI normally leaves the option omitted so it follows
`--max-context`; the distinction matters primarily to a concurrent Engine or server.

At Engine startup GInfer reserves model weights, persistent sequence state, one phase-reused
Program scratch arena, the maximum Vision request-transient buffer when Vision is enabled, and a
separate CUDA Graph driver allowance. Scratch is the maximum of the enabled Text, MTP, DFlash, and
Vision phases, not their sum. Its prefill bound uses
`min(--prefill-chunk,--max-context)`. The request-transient buffer is also frozen at startup; a
media request activates only the needed prefix and performs no project-owned device allocation or
growth.

All weight, sequence, workspace, request-transient, and graph allocations are released when the
Engine is destroyed.
