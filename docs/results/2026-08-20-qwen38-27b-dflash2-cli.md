# Qwen3.8-27B DFlash2 CLI on RTX 5090 and RTX 3090

Date: 2026-08-20, updated 2026-08-21. These measurements qualify the registered
`qwen3.8-27b/groupwise-int-dflash2-q4`, `qwen3.8-27b/groupwise-int-dflash2-w8`,
`qwen3.8-27b/nvfp4-dflash2-q4`, and `qwen3.8-27b/nvfp4-dflash2-w8` artifacts through the public
GInfer Engine route. The RTX
5090 runs use a Release `sm_120a` image built with CUDA 13.3. The RTX 3090 runs use a Release
`sm_86` image built with CUDA 13.1.

These are execution measurements, not model-quality results. Q4 DFlash2, W8 DFlash2, MTP, and
autoregressive decode can follow different sampled trajectories under the same seed, so their
complete-run ratios are not isolated storage- or kernel-format speedups.

## Matched 999/1,000 production workload

All GInfer rows use the same 999-token messages prompt, 1,000 generated tokens, thinking disabled,
INT8 group-64 KV, a 16,384-token context, 1,024-token prefill chunks, and seed 42. Sampling is
temperature 0.7, top-p 0.80, top-k 20, min-p 0, presence penalty 1.5, and frequency penalty 0.
The default `--spec auto` policy selects DFlash2 with four draft tokens for either companion
artifact.

### RTX 5090

| Startup profile | Companion storage | Prefill tok/s | Decode tok/s | Overall tok/s | Resident weights | Speculative result |
|---|---|---:|---:|---:|---:|---|
| AR, explicit `--spec none` | nonresident | 3,154.64 | 82.58 | 80.55 | 15.92 GiB | off |
| DFlash2 `k=4`, default auto | native Q4G64 | 3,114.89 | 134.85 | 129.38 | 16.87 GiB | 590/1,635 accepted (36.09%); 2.44 tok/round; 409 rounds |
| DFlash2 `k=4`, default auto | W8G32 | 3,115.91 | 138.26 | 132.51 | 17.82 GiB | 613/1,539 accepted (39.83%); 2.59 tok/round; 385 rounds |
| MTP3 + optimized proposal head | DFlash nonresident | 3,108.71 | 158.12 | 150.62 | 16.67 GiB | 595/1,212 accepted (49.09%); 2.47 tok/round; 404 rounds |

The NVFP4/FP8 backbone was also repeated twice on 2026-08-21 under eager CUDA module loading:

| Startup profile | Prefill runs / mean tok/s | Decode runs / mean tok/s | Overall mean tok/s | Resident weights | Speculative result |
|---|---:|---:|---:|---:|---|
| Companion-free `nvfp4`, MTP3 | 7,718.03, 7,794.80 / **7,756.42** | 144.12, 143.55 / **143.84** | **141.36** | MTP profile | 582/1,251 accepted (46.52%); 2.40 tok/round |
| `nvfp4-dflash2-q4`, DFlash2 `k=4` | 7,802.41, 7,629.25 / **7,715.83** | 154.30, 154.56 / **154.43** | **151.55** | 19.93 GiB | 633/1,464 accepted (43.24%); 2.73 tok/round; 366 rounds |
| `nvfp4-dflash2-w8`, DFlash2 `k=4` | 7,795.87, 7,557.10 / **7,676.49** | 142.82, 141.23 / **142.02** | **139.58** | 20.88 GiB | 617/1,525 accepted (40.46%); 2.62 tok/round; 382 rounds |

Each profile reproduced its complete output and acceptance trajectory exactly across both runs.
The Q4 companion is the production choice for this backbone: it decodes 8.7% faster than Q8,
uses 1,022,197,760 fewer resident bytes, and is 7.4% faster than the MTP3 control on this complete
sampled request. The Q4 and Q8 rows follow different trajectories, so this is not an isolated
kernel-format ratio or a model-quality comparison.

Within the groupwise rows, the production-default Q4 DFlash2 route is 63.3% faster than AR decode
on this complete request while adding 0.95 GiB of resident companion weights. The groupwise W8 row
is 2.5% faster than Q4 but uses another 1,022,197,760 device bytes (0.95 GiB) and follows a
different accepted trajectory. Groupwise MTP3 remains faster on this particular instruct sample
because its acceptance is higher and its round is lighter; DFlash2 is nevertheless the default
companion-artifact policy and supports the published low-VRAM Q4 storage profile.

The two DFlash rows used 165.63 MiB of workspace against a 172.57 MiB planned capacity. Their CUDA
Graph profiles were enabled through the ordinary CLI default, including the formerly untested
default-startup path.

### RTX 3090

The physical Ampere run used the Q4 companion and the same request/options:

| Engine and route | Prefill tok/s | Decode tok/s | Overall tok/s | Speculative result |
|---|---:|---:|---:|---|
| GInfer `sm_86`, Q4 DFlash2 `k=4` (production default) | 954.06 | 53.82 | 50.99 | 594/1,617 accepted (36.73%); 2.47 tok/round; 405 rounds |
| GInfer `sm_86`, Q4 DFlash2 `k=3` (explicit) | 955.82 | 60.30 | 56.78 | 578/1,257 accepted (45.98%); 2.38 tok/round; 420 rounds |
| llama.cpp b10523 `d59d455fd`, Q4_K_M + in-model MTP3 | 863.2 | 50.1 | not reported | stock CLI summary did not print acceptance |

This is a complete-route comparison, not a DFlash2 implementation comparison: the stock llama.cpp
revision does not include the proposed DFlash2 support from ggml-org/llama.cpp PR 27342 and
therefore runs its in-model MTP drafter. GInfer physically loaded the Q4 companion, executed real
DFlash2 rounds with the default CUDA Graph configuration, and completed the 1,000-token request on
the RTX 3090. That qualifies this exact Q4 DFlash2 route on `sm_86`; it does not qualify the W8
profile, Muse's other storage profiles, or any `sm_89` route without their own physical runs.

An initial matched sweep before the final Ampere compact-kernel retune measured decode rates of
49.06, 56.60, 59.67, 50.33, 50.33, 44.49, and 43.94 tok/s for learned DFlash2 widths `k=1..7`.
The same run measured 64.31 tok/s for MTP3. The final retained kernels move the production `k=4`
route from 50.33 to 53.82 tok/s (+6.9%) and `k=3` from 59.67 to 60.30 tok/s (+1.1%). This one
sample favors `k=3`, but the sampled trajectory and workload are not a sufficient basis for
changing the cross-hardware automatic default from four.

## Adaptive context-copy verification through position 15

Qwen3.8's five-layer companion remains a learned `k<=7` drafter. GInfer now independently indexes
the request's represented token ledger for an exact six-to-twelve-token suffix match and can use
the earlier occurrence's represented continuation as a point-mass proposal. Rejection sampling
remains lossless because `q` describes the exact token handed to target verification. After two
fully accepted context-copy rounds for every row in the compact batch, target verification expands
to at most 15 proposals; the learned DFlash2 pass stays at its configured width. Batch one has a
dedicated wide CUDA Graph, while expanded batches two through eight execute eagerly.

A physical RTX 3090 copy-workload check used learned `k=3`, a 121-token repeated-document prompt,
and 64 sampled output tokens. It reached 113.70 decode tok/s and 85.25 overall tok/s over nine
rounds. Three rounds expanded, 54 of 63 proposals were accepted, and the per-position counts were
`9,8,8,3,3,3,3,3,2,2,2,2,2,2,2`; the nonzero counts through position 15 directly prove that the
full wide verification/accept/commit path executed. This is a targeted copy-path qualification,
not a general-production throughput comparison with the 999/1,000 workload above.

CLI, serving request logs, and benchmark schema 12 now report the configured learned
`draft_window`, maximum `verification_window`, `expanded_rounds`, and all fifteen per-position
acceptance counters separately.

## Compact Q4 kernel work

The Q4 implementation has closed small-token schedules for the exact Qwen DFlash2 geometries on
`sm_120a`. The retained `sm_86` retune also adds exact compact attention/GDN projection schedules,
a direct split-4 Q5 route through `T=8`, fused GDN convolution snapshots at `T=7,8`, and a 32-row
interleaved Q4 SwiGLU schedule. Independent decoded-weight FP64/FP32 oracles cover every changed
shape and the focused suites pass on the physical RTX 3090.

Representative cold-cache RTX 5090 operator measurements at `T=5` were:

| Logical matrix `[N,K]` | Before | Final route | Change |
|---|---:|---:|---:|
| `[4096,5120]` | 24.0 us | 15.8 us | 34% lower |
| `[17408,5120]` | 83.36 us | 44.29 us | 47% lower |
| `[5120,4096]` | 19.49 us | 12.8 us | 34% lower |

The large `[5120,17408]` and `[5120,25600]` routes cross over later and therefore retain SIMT for
`T<=8`, using the closed schedule only at `T=9..16`. Small selector and grouped-convolution
projections remain on SIMT because the closed schedule was slower there. This selective dispatch
is the measured implementation; no route is retained merely because it exists.

The RTX 3090 Nsight Systems trace attributes most DFlash2 time to the target Q4/Q5 verification
backbone. The two W8 vocabulary heads account for about 5.2% of GPU time, while split verification
attention and selector/accept together are below the dominant Linear kernels. GInfer already uses
split-KV verification attention; further Ampere gains should therefore target the compact target
Linear schedules and then the two vocabulary heads rather than reworking the selector first.

## Artifact and residency facts

| Profile | Artifact bytes | DFlash-resident device bytes | Difference from Q4 |
|---|---:|---:|---:|
| `groupwise-int-dflash2-q4` | 19,233,276,416 | 18,116,223,488 | 0 |
| `groupwise-int-dflash2-w8` | 20,255,474,176 | 19,138,421,248 | +1,022,197,760 |
| `nvfp4-dflash2-q4` | 22,515,440,128 | 21,398,354,432 | 0 |
| `nvfp4-dflash2-w8` | 23,537,637,888 | 22,420,552,192 | +1,022,197,760 |

Both groupwise artifacts carry the same 1,118-tensor base inventory plus the exact 81-tensor DFlash2 companion.
With `--spec none`, both use the same 17,093,490,688-byte base residency. There is no registered
companion-free Qwen3.8 groupwise identity.

The two NVFP4 companion artifacts carry the registered 1,118-tensor mixed NVFP4/FP8 base plus the
same exact 81-tensor DFlash2 companion. With `--spec none`, both use the same
20,375,621,632-byte base residency. The companion-free `nvfp4` identity remains registered for MTP,
Vision, and autoregressive startup.

## Reproduction commands

`MESSAGES` is the messages JSON used by the earlier paired Qwen3.8 CLI comparison; it tokenizes to
999 prompt tokens with the GInfer frontend.

```bash
GINFER=./build-120a/apps/ginfer
MESSAGES=/tmp/cmp-p1000.json
COMMON="--messages $MESSAGES --max-new 1000 --max-context 16384 \
  --kv-dtype int8 --kv-capacity auto --prefill-chunk 1024 --no-thinking \
  --temperature 0.7 --top-p 0.80 --top-k 20 --min-p 0 \
  --presence-penalty 1.5 --seed 42"

# Production default: auto resolves the Q4 companion to DFlash2 k=4.
$GINFER out/qwen3_8_27b_dflash2_q4.ginfer $COMMON

# Explicit learned k=3 row used in the physical Ampere comparison.
$GINFER out/qwen3_8_27b_dflash2_q4.ginfer $COMMON --spec dflash --draft-tokens 3

# Matched W8 companion.
$GINFER out/qwen3_8_27b_dflash2_w8.ginfer $COMMON

# Mixed NVFP4/FP8 backbone with Q4 or Q8 DFlash2. Both identities auto-select k=4.
$GINFER out/qwen3_8_27b_nvfp4_dflash2_q4.ginfer $COMMON
$GINFER out/qwen3_8_27b_nvfp4_dflash2_q8.ginfer $COMMON

# Explicit baselines against the same complete Q4 artifact.
$GINFER out/qwen3_8_27b_dflash2_q4.ginfer $COMMON --spec none
$GINFER out/qwen3_8_27b_dflash2_q4.ginfer $COMMON \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

The RTX 3090 GInfer command changes only the image and artifact path:

```bash
./build-86/apps/ginfer models/qwen3_8_27b_dflash2_q4.ginfer $COMMON
```

The physical width-15 context-copy qualification used this exact command (the literal prompt is
intentional):

```bash
./build-86/apps/ginfer models/qwen3_8_27b_dflash2_q4.ginfer \
  --prompt "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango alpha bravo charlie delta echo foxtrot" \
  --max-context 2304 --kv-capacity 2304 --prefill-chunk 1024 --max-new 64 \
  --spec dflash --draft-tokens 3 \
  --temperature 0.7 --top-p 0.80 --top-k 20 --min-p 0 \
  --presence-penalty 0 --frequency-penalty 0 --seed 42 --raw-output
```

The stock llama.cpp comparison command was:

```bash
./build-cuda13/bin/llama-cli \
  -m /path/to/Qwen3.8-27B-Q4_K_M.gguf \
  -f /tmp/cmp-p1000.txt -n 1000 -c 16384 -ngl 99 \
  -ctk q8_0 -ctv q8_0 \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.0 \
  --presence-penalty 1.5 --repeat-penalty 1.0 \
  --jinja --reasoning off --reasoning-budget 0 \
  --conversation --single-turn --no-display-prompt
```
