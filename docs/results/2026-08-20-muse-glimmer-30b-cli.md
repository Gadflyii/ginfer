# Muse Glimmer 30B CLI on RTX 5090 and RTX 3090

Date: 2026-08-20. Host `Ron-9950X3D2` under WSL, NVIDIA GeForce RTX 5090, Release
`sm_120a` image built with CUDA 13.3. The measured artifacts are the registered
`muse-glimmer-30b/groupwise-int` BF16-DFlash profile and
`muse-glimmer-30b/groupwise-int-dflash-q4` low-VRAM profile, plus the RedHat-backed
`muse-glimmer-30b/nvfp4` target with the same Q4 DFlash2 companion.

These are CLI execution measurements, not model-quality results. The three DFlash storage profiles
can take different sampled trajectories even with the same seed, so their single-run throughput
ratios are not isolated kernel-speed measurements.

## Matched 1,000-token workload

All rows use the same 1,050-token message prompt, 1,000 generated tokens, high reasoning strength,
INT8 group-64 KV, a 16,384-token context, 1,024-token prefill chunks, and seed 42. Sampling is
temperature 1.0, top-p 0.95, top-k 64, min-p 0, with zero presence and frequency penalties.
DFlash uses the best sampled window from the short sweep below: four draft tokens (`k=4`). CUDA
Graph capture is enabled for AR but is not yet implemented for the Muse DFlash round.

| Mode | DFlash storage | Prefill tok/s | Decode tok/s | Overall tok/s | Resident weights | Acceptance |
|---|---|---:|---:|---:|---:|---|
| AR | not resident | 3,237.42 | **75.54** | **73.80** | 15.11 GiB | off |
| DFlash2 | BF16 conv/selector | 890.95 | **90.74** | **82.05** | 16.78 GiB | 645/1,416 (45.55%); 2.82 tok/round; 354 rounds |
| DFlash2 | native Q4G64 conv/selector | 1,095.66 | **99.12** | **90.61** | 16.49 GiB | 666/1,331 (50.04%); 3.00 tok/round; 333 rounds |
| DFlash2 | RedHat NVFP4 backbone + native Q4G64 companion | 2,167.34 | **112.98** | **107.22** | 17.22 GiB | 660/1,353 (48.78%); 2.95 tok/round; 339 rounds |

The prefill columns in this original table include CUDA's former lazy module materialization in the
first CLI request and are cold-start observations, not isolated prefill-kernel comparisons. This
was especially misleading for the two groupwise profiles: their Text backbone and every DFlash
matrix exercised while seeding the prompt cache have identical storage; the 13 BF16-versus-Q4
matrices are used only after decode begins.

On 2026-08-21, four fresh Q4-DFlash processes with lazy loading measured 1,456.92--1,463.94
prefill tok/s. Selecting eager module loading moved about 0.40 seconds into the separately reported
Engine-construction phase. The same complete 1,050/1,000 command then measured 2.889 seconds for
Engine construction, 0.341 seconds / **3,080.31 tok/s** for prefill, 9.650 seconds /
**103.53 tok/s** for decode, and **100.09 tok/s** overall. A one-output control measured AR at
3,133.53 prefill tok/s and Q4-DFlash at 3,080.48 tok/s, a 1.69% complete-route difference. GInfer
now selects eager CUDA module loading before creating the device context so one-time code loading
belongs to startup rather than request timing; total cold-start cost is not hidden.

The native-NVFP4 profile was then repeated twice with the same corrected startup policy and the
same complete 1,050/1,000 request. Prefill measured **9,559.93** and **9,512.16 tok/s**, decode
measured **111.57** and **111.34 tok/s**, and overall throughput measured **110.33** and
**110.10 tok/s**. Both runs reproduced the exact 660/1,353 accepted trajectory, 2.95 tokens per
round, and 302.44 MiB workspace high-water. The original table's 2,167.34 NVFP4 prefill value is
therefore also a lazy-loading-contaminated cold observation; the corrected two-run mean is
**9,536.05 prefill tok/s**, **111.46 decode tok/s**, and **110.22 overall tok/s**.

The Q4 row is 9.24% faster than the BF16 row for this one full seed-42 trajectory. Its acceptance
length is also higher, so that percentage must not be generalized as a storage-format speedup.
Both groupwise DFlash rows reached the 1,000-token output limit. The Q4 artifact's AR execution is
identical to the default artifact's AR execution because the 13 profile-specific matrices are
DFlash-only and remain nonresident without `--spec dflash`.

The NVFP4 row also reached the 1,000-token output limit. It is the highest-throughput complete run
in this table despite a lower accepted-prefix length than the groupwise Q4 row, but the three
profiles still follow different logits and sampled trajectories. The table establishes the
production request result, not an isolated NVFP4-versus-Q4 kernel ratio. It is the representative
median-decode run from the post-retune repetitions below, with `k=4` held constant as the profiling
control; the other rows are their original single runs.

## Native-NVFP4 decode-kernel and selector retune

The native-NVFP4 route was profiled with four DFlash draft tokens held constant so kernel changes
could be compared on one workload. This does not specialize the implementation to `k=4`: no
speculation remains the zero-width mode, and DFlash accepts 1--15 drafts, producing target verify
widths 2--16. The final route keeps query/gate and key/value projections on A16 through seven
tokens, moves attention and MLP output projections to W4A4 at five tokens, and uses dedicated
small-token W4A4 schedules for the two Muse MLP geometries. An independent decoded-NVFP4 FP64
oracle covers the affected real shapes at the route and tile boundaries.

The same profile exposed a separate DFlash2 selector bottleneck shared by every Muse storage
profile: each draft column ranked all 202,048 unary logits in one CTA, then one thread evaluated
the selector path. The replacement constructs the exact top 16 with a caller-owned hierarchical
partial/group workspace and scores each candidate with a warp. It preserves descending-logit and
lower-token-id tie ordering, the sequential predecessor dependency, and the existing sampling
distribution. The production route handles every configured draft width from 1 through 15; `k=4`
is only the controlled profile point used here.

Five complete repetitions of the matched 1,000-token seed-42 request produced. The final two were
run after the selector's cross-CTA publication fence and maximum-width regression were relinked:

| Repeat | Prefill tok/s | Decode tok/s | Overall tok/s | Rounds | Accepted / drafted |
|---:|---:|---:|---:|---:|---:|
| 1 | 2,151.44 | 113.02 | 107.21 | 339 | 660 / 1,353 |
| 2 | 2,167.34 | 112.98 | 107.22 | 339 | 660 / 1,353 |
| 3 | 2,175.62 | 113.35 | 107.57 | 339 | 660 / 1,353 |
| 4 | 1,368.99 | 110.39 | 101.87 | 339 | 660 / 1,353 |
| 5 | 1,523.95 | 111.86 | 103.95 | 339 | 660 / 1,353 |

Decode averaged 112.32 tok/s and ranged from 110.39 to 113.35 tok/s. The accepted trajectory was
identical in all three repetitions. Prefill varied more and is reported rather than interpreted;
the changed schedules target compact decode widths, not the 1,024-token prefill route.

On the exact cold-cache decode geometries, the W4A4 MLP-down schedule improved from 65.9 to 56.6
us at five tokens, while dispatch moved that projection from the slower 85.3 us A16 route to W4A4.
The paired MLP-input projections improved from 55.4 to 52.5 us each. The final standalone sum of
the target-layer Linear calls is 249.9 us versus 298.7 us before the retune (16.4% lower); this is
an operator sum, not an end-to-end percentage.

Matched Nsight Systems runs used the same 1,050-token prompt and 256-token generation request.
Because the numerical route change produced a different sampled trajectory, raw short-run tok/s
is not an isolated comparison. Normalizing by completed DFlash rounds gives 29.30 ms/round before
and 27.65 ms/round after the retune, a 5.61% reduction. The final profile still identifies the
NVFP4 MLP-input projection as memory-bound (about 1.33 TB/s and 74.5% DRAM throughput); larger
gains now require a higher-level fusion or materially better memory reuse rather than another
launch-shape tweak.

At `k=4`, the unary top-k stage fell from 965.45 to 53.93 us per round and the path stage from
267.60 to 16.93 us. The complete selector therefore fell from 1.233 ms to 70.86 us per round
(17.4x, 94.3% lower). A `k=15` endpoint profile measured about 177.4 us per round, confirming that
the hierarchy covers the configured maximum rather than specializing the four-draft case. These
are operator-level profile sums; the complete-request rows above are the end-to-end evidence.

## Draft-window and repeated-seed evidence

The sampled window sweep uses the same 1,050-token prompt and sampling settings with a 256-token
output limit. Values are mean decode throughput over seeds 42, 43, and 44 for the Q4 profile. It
predates the native-NVFP4 kernel retune and is retained to support the measured `k=4` recommendation;
speculative-width performance will be swept again separately.

| Draft tokens | Mean decode tok/s |
|---:|---:|
| 4 | **113.04** |
| 5 | 107.88 |
| 7 | 100.37 |
| 9 | 93.12 |
| 11 | 96.88 |
| 15 | 94.13 |

At `k=4`, the individual BF16 results were 111.12, 114.81, and 103.36 tok/s (mean 109.76),
while Q4 produced 104.11, 117.56, and 117.45 tok/s (mean 113.04). Mean acceptance lengths were
3.41 and 3.42 tokens/round respectively. This representative three-seed sample places Q4 2.99%
above BF16, with nearly matched acceptance length; it is still a small performance sample and not
a quality comparison.

The NVFP4 profile produced 118.67, 104.04, and 109.36 tok/s for seeds 42, 43, and 44 (mean
110.69). Its acceptance rates were 61.43%, 49.42%, and 53.56%, with a mean accepted length of
3.18 tokens/round. That mean lies between the BF16 and groupwise-Q4 results; the per-seed spread
again shows why the 1,000-token row should be read as an end-to-end production result rather than a
format-only speed ratio.

For a separate greedy 256-token stress run, `k=15` reached 205.50 decode tok/s with
seed-independent output. Greedy and stochastic results are distinct workloads and should not be
compared as if they measured the same acceptance distribution.

The isolated warm grouped-dynamic-convolution benchmark at batch 1, draft width 15, and all ten
prepares measured a 326.40 us BF16 median (32.64 us per prepare) and a 241.65 us native-Q4 median
(24.16 us per prepare), a 1.35x operator-level improvement. This benchmark covers the projection,
BF16 dynamic boundary, and grouped convolution; it does not establish an end-to-end speedup by
itself.

## Full-context retrieval ladder and llama.cpp DFlash1

The same RTX 5090 ran the committed 8K, 64K, and 128K retrieval fixtures through GInfer DFlash2
and llama.cpp DFlash1. Both runners use a 131,072-token context, INT8 KV, high reasoning strength,
four draft tokens, temperature 1.0, top-p 0.95, top-k 64, min-p 0, zero penalties, and seed 42.
The output limits are 512, 512, and 2,032 tokens respectively; every run terminated naturally
before its limit.

The complete GInfer ladder used the native `groupwise-int-dflash-q4` artifact; the NVFP4 profile
was repeated at the 128K edge. llama.cpp build
`b354-d59d455fd` used `Muse-Glimmer-30B-UD-Q4_K_XL.gguf` with the DFlash1
`dflash-kquant.gguf` sidecar. These are different storage and speculative implementations. More
importantly, their GGUF and `.ginfer` frontends/templates tokenize the same raw message fixture
differently, so the rows compare complete production-style routes rather than an identical token
stream or isolated kernel.

| Fixture | Engine | Prompt tokens | Output tokens | Prefill tok/s | Decode tok/s | Acceptance | Retrieval result |
|---|---|---:|---:|---:|---:|---|---|
| 8K | GInfer DFlash2 Q4 | 7,641 | 343 | 2,356.56 | 117.31 | 67.20%; 3.69 tok/round | exact |
| 8K | llama.cpp DFlash1 Q4_K_XL | 7,457 | 164 | 2,258.03 | 142.76 | 62.23%; 3.49 tok/round | exact |
| 64K | GInfer DFlash2 Q4 | 64,059 | 223 | 2,598.41 | 117.13 | 70.69%; 3.83 tok/round | exact |
| 64K | llama.cpp DFlash1 Q4_K_XL | 62,521 | 105 | 3,077.34 | 157.78 | 80.00%; 4.20 tok/round | exact |
| 128K | GInfer DFlash2 Q4 | 129,041 | 854 | 2,063.61 | 102.74 | 68.23%; 3.73 tok/round | semantic; spaced digits |
| 128K | GInfer DFlash2 NVFP4/Q4 | 129,041 | 223 | 3,681.28 | 108.74 | 67.50%; 3.70 tok/round | exact |
| 128K | llama.cpp DFlash1 Q4_K_XL | 125,885 | 198 | 2,832.34 | 127.35 | 61.40%; 3.46 tok/round | exact |

Both complete routes retrieved the correct recovery code and color at every rung. GInfer's 128K response
rendered the correct code as `4 9 3 8 1 7`, missing the fixture's exact no-space format; llama.cpp
returned `493817`. GInfer allocated the full 131,072-token INT8 KV capacity: 3.35 GiB KV payload,
3.39 GiB total sequence reservation, 20.16 GiB planned device total, and 10.04 GiB free after
startup. No nonfinite attention output or high-context execution failure recurred after the
sliding-attention empty-tile correction.

The native-NVFP4 edge run used the same GInfer fixture and command surface and returned the exact
recovery line `ORCHID=493817; COLOR=COBALT`. It allocated the complete 131,072-token cache with
17.22 GiB of resident weights, a 3.39 GiB sequence reservation, a 20.91 GiB planned total, and
9.29 GiB free after startup. Its higher prefill result is a complete-route measurement of this
profile, not a claim about the RadixArk mixed checkpoint or an isolated Linear kernel.

The reproducible runners preserve the complete expanded launch command and combined output under
`profiles/bench/muse-glimmer-context-ladder/<run-label>/`:

```bash
RUN_LABEL=20260820-ginfer-dflash2-context \
  ./bench/run_muse_glimmer_context_ladder_ginfer.sh

RUN_LABEL=20260820-llamacpp-dflash1-context \
  ./bench/run_muse_glimmer_context_ladder_llamacpp.sh

GINFER_MODEL=out/muse_glimmer_30b_nvfp4.ginfer \
RUN_LABEL=20260820-ginfer-nvfp4-context \
  ./bench/run_muse_glimmer_context_ladder_ginfer.sh 128k
```

### Native NVFP4 with 15 draft tokens

The complete native-NVFP4 ladder was repeated with `--draft-tokens 15`; every other setting and
fixture remained unchanged. This is a production-sampling run, not the separate greedy timing
case above.

| Fixture | Output tokens | Prefill tok/s | Decode tok/s | Acceptance | Retrieval result |
|---|---:|---:|---:|---|---|
| 8K | 204 | 4,649.79 | 83.98 | 26.51%; 4.98 tok/round | exact |
| 64K | 121 | 5,536.92 | 34.93 | 32.12%; 5.82 tok/round | exact |
| 128K | 389 | 3,545.63 | 18.32 | 26.24%; 4.94 tok/round | semantic; answer repeated in merged transcript |

The 128K `k=15` run remained finite, stayed within the same 302.44 MiB workspace high-water, and
retrieved `493817` and `COBALT`. Its merged stdout/stderr transcript places the exact content answer
immediately after the reasoning channel's final answer text, so the combined log contains the line
twice. The test runner intentionally preserves both channels; this is not an attention or cache
failure.

At 128K, `k=15` commits 4.94 tokens per target round versus 3.70 for `k=4`, but decode falls from
108.74 to 18.32 tok/s (5.94x slower). Each round verifies 16 target positions instead of five, and
the extra accepted suffix does not amortize that long-context attention work. The sampled outputs
also differ in length, so this is a complete-route comparison rather than an isolated kernel ratio,
but the margin is sufficient to reject `k=15` as the production long-context setting. `k=4`
remains the measured recommendation.

The GInfer runner exposes the width while retaining four as its default:

```bash
DFLASH_DRAFT_TOKENS=15 \
GINFER_MODEL=out/muse_glimmer_30b_nvfp4.ginfer \
RUN_LABEL=20260820-ginfer-nvfp4-dflash15-context \
  ./bench/run_muse_glimmer_context_ladder_ginfer.sh
```

## RTX 3090 native-Q4 qualification

A physical NVIDIA GeForce RTX 3090 ran the same 1,050/1,000-token Q4 DFlash2 production workload
through a Release `sm_86` image built with CUDA 13.1. The final selective Ampere dispatch keeps
small Muse matrices on SIMT and uses the existing Q4 MMA route only for the three large compact
decode geometries where it wins.

| Profile | Prefill tok/s | Decode tok/s | Overall tok/s | Acceptance |
|---|---:|---:|---:|---|
| native Q4G64 DFlash2 `k=4` | 534.21 | 32.57 | 30.64 | 657/1,365 (48.13%); 2.92 tok/round; 342 rounds |

The request reached its 1,000-token output limit with 16.49 GiB of resident weights and a
280.50 MiB workspace high-water. The same artifact also passed the real Engine regression and the
independent decoded-Q4 FP64 operator oracle on that device. This qualifies the native-Q4 DFlash2
route on `sm_86`; it does not qualify the BF16 companion or any Muse `sm_89` route.

Cold-cache operator measurements explain the selective dispatch. At five compact columns, the
large `[19968,6656]`, `[6656,19968]`, and `[6656,33280]` matrices improved by 5.7%, 3.1%, and 3.7%
respectively. At nine or fifteen columns the MMA route wins by much more, while applying it to the
smaller Muse matrices was 15% to 3.6x slower and was therefore rejected. The Nsight Systems trace
attributes 72.3% of the pre-change DFlash2 GPU time to generic Q4 SIMT Linear calls; attention,
selector, and acceptance are not the dominant Ampere bottlenecks.

The launch command is identical to the Q4 command below except for the architecture-specific
binary:

```bash
./build-86/apps/ginfer models/muse_glimmer_30b_dflash_q4.ginfer \
  --messages /tmp/cmp-p1000.json \
  --max-new 1000 --max-context 16384 --kv-dtype int8 --kv-capacity auto \
  --prefill-chunk 1024 --reasoning-effort high \
  --spec dflash --draft-tokens 4 \
  --temperature 1.0 --top-p 0.95 --top-k 64 --min-p 0 \
  --presence-penalty 0 --frequency-penalty 0 --seed 42
```

## Memory and artifact profile

The default DFlash-resident device arena is 18,019,114,496 bytes. The native Q4 profile is
17,701,999,104 bytes, saving 317,115,392 bytes (302.42 MiB). Its DFlash component is
1,473,220,096 bytes instead of 1,790,335,488 bytes. AR-only residency remains exactly
16,228,779,008 bytes for both identities.

The Q4 profile stores ten grouped-convolution projection matrices and the selector hidden
projection plus two codebooks directly in GInfer's native Q4G64 row-split format. It performs no
runtime repacking. This is not GGUF `Q4_K_M`, and no numerical or task-quality equivalence to a
GGUF artifact is claimed. The real-artifact Engine regression executes a DFlash round through the
Q4 convolution and selector paths; the focused operators are also qualified against independent
decoded-weight mathematical oracles.

The native-NVFP4 artifact is 18,516,895,232 bytes. Its DFlash-resident device arena is
18,488,472,064 bytes: 786,472,960 bytes (750.04 MiB) above the groupwise Q4 profile and
469,357,568 bytes (447.61 MiB) above the BF16-companion profile. This profile preserves 416 RedHat
transformer-block matrices in native NVFP4 and retains the low-memory Q4 companion. The 208 FP32
activation divisors are validated host values rather than resident device tensors.

The groupwise DFlash rows used 280.50 MiB of the 286.74 MiB workspace arena. The NVFP4 row used
302.44 MiB of a 308.68 MiB arena because W4A4 prefill owns caller-planned activation scratch. The
Muse DFlash schedule remains uncaptured, so its reported CUDA Graph allowance is zero.

## RedHat and RadixArk comparison target

The registered profile is based on
[RedHatAI/Muse-Glimmer-30B-NVFP4](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-NVFP4), which
places all 416 transformer-block linears in NVFP4 and leaves the embedding and output head in their
original precision. [RadixArk/Muse-Glimmer-NVFP4](https://huggingface.co/RadixArk/Muse-Glimmer-NVFP4)
is not the same allocation: at its measured repository revision it stores 312 decoder matrices and
the embedding in NVFP4, the 52 MLP down projections and output head in block-32 MXFP8, and all 52
value projections in BF16. The current files therefore also differ from the model card's prose
claim that value projections are MXFP8.

The production comparison will retain the same raw messages, sampling, output budgets, DFlash
window, and context ladder used above. It needs two separately reported views: backbone-only AR to
compare the quantized targets without acceptance effects, and DFlash production mode to compare the
complete deployed routes. GInfer does not yet implement RadixArk's block-32 MXFP8 storage or Linear
path, so no RadixArk number is reported here. Recasting it to row-scaled FP8 or labeling it as the
RedHat `nvfp4` identity would change the checkpoint and invalidate the intended comparison.

## Commands

```bash
# AR (the Q4-only DFlash matrices remain nonresident)
./build-120a/apps/ginfer out/muse_glimmer_30b_dflash_q4.ginfer \
  --messages /tmp/cmp-p1000.json \
  --max-new 1000 --max-context 16384 --kv-dtype int8 --kv-capacity auto \
  --prefill-chunk 1024 --reasoning-effort high \
  --spec none \
  --temperature 1.0 --top-p 0.95 --top-k 64 --min-p 0 \
  --presence-penalty 0 --frequency-penalty 0 --seed 42

# Default BF16-DFlash profile
./build-120a/apps/ginfer models/muse_glimmer_30b.ginfer \
  --messages /tmp/cmp-p1000.json \
  --max-new 1000 --max-context 16384 --kv-dtype int8 --kv-capacity auto \
  --prefill-chunk 1024 --reasoning-effort high \
  --spec dflash --draft-tokens 4 \
  --temperature 1.0 --top-p 0.95 --top-k 64 --min-p 0 \
  --presence-penalty 0 --frequency-penalty 0 --seed 42

# Native low-VRAM Q4-DFlash profile
./build-120a/apps/ginfer out/muse_glimmer_30b_dflash_q4.ginfer \
  --messages /tmp/cmp-p1000.json \
  --max-new 1000 --max-context 16384 --kv-dtype int8 --kv-capacity auto \
  --prefill-chunk 1024 --reasoning-effort high \
  --spec dflash --draft-tokens 4 \
  --temperature 1.0 --top-p 0.95 --top-k 64 --min-p 0 \
  --presence-penalty 0 --frequency-penalty 0 --seed 42

# RedHat native-NVFP4 backbone with Q4-DFlash companion
./build-120a/apps/ginfer out/muse_glimmer_30b_nvfp4.ginfer \
  --messages /tmp/cmp-p1000.json \
  --max-new 1000 --max-context 16384 --kv-dtype int8 --kv-capacity auto \
  --prefill-chunk 1024 --reasoning-effort high \
  --spec dflash --draft-tokens 4 \
  --temperature 1.0 --top-p 0.95 --top-k 64 --min-p 0 \
  --presence-penalty 0 --frequency-penalty 0 --seed 42
```
