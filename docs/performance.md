# Single-GPU serving performance

Tested Git revisions:

- Qwen3.8-27B NVFP4 MTP0 context-length serving:
  `f08597d6eaafce5b875934aaa85854fcd5426df8`;
- Qwen3.8-27B NVFP4 MTP3 single-request and concurrent fixed-corpus serving:
  `32c9881b6783949df4999422a764b3dcaa111b13`;
- Concurrent MTP3 decode saturation for the three measured Qwen3.6 artifact profiles:
  `26da9df7c1b3d3c04ea7bbd730271aa01d00742a`;
- Refreshed Qwen3.6-35B-A3B and Qwen3.6-27B NVFP4 MTP3:
  `f4f21cc36bd1a83cbc046f668719d591dc9c1e2e`;
- Qwen3.6-35B-A3B stored MTP3 response audit:
  `b1a220f028aa750f75bceb3522ac00bbaab7e42d`;
- Qwen3.6-35B-A3B DFlash block=8 (`k=7`):
  `0dc94097e8ec5c5bcf59b9e13e9d1852f504eb61`;
- Qwen3.6-27B NVFP4 accuracy and MTP0:
  `b3d4d0f50b868711c62432bbd68e746217a2f49a`;
- Qwen3.6-27B groupwise-int MTP3: `5ea3242a206cdb0c4c1beaeb9d8a3048e6248423`;
- Qwen3.6-35B-A3B MTP0 and Qwen3.6-27B groupwise-int MTP0:
  `0795169393cab0f2c16246d4bac20dee735dc2a4`;
- RTX PRO 6000 Blackwell Workstation Edition (`sm_120a`, 188 SMs) Path 1/2 CLI smoke and
  `long_niah_8k` subset: this tree after Tasks 1–5 (`159b8c9c`) plus the dual-SKU and measurement
  documentation commits;
- RTX 3090 (`sm_86`, 82 SMs) Path 1 groupwise-int CLI smoke and 24 GB envelope: this tree
  (`feat/sm-t8`);
- RTX 4090 (`sm_89`, 128 SMs) Path 1 groupwise-int CLI smoke and 24 GB envelope: this tree
  (`feat/sm-t7`);
- Qwen3.8-27B CLI vs ggml-org llama.cpp MTP3 instruct (4090 1k/1k): this tree after the 131k
  INT8 docs (`c50aa79e`);
- Muse Glimmer 30B RTX 5090 CLI AR, BF16/Q4-DFlash2, and RedHat native-NVFP4/Q4-DFlash2 storage
  profiles, matched 1050/1000 INT8 seed-42 rows plus a three-seed 256-token comparison and 128K
  retrieval:
  `docs/results/2026-08-20-muse-glimmer-30b-cli.md` (this tree).
- Qwen3.8-27B groupwise and NVFP4-backbone Q4/Q8 DFlash2, AR, and MTP3 matched 999/1000 CLI on
  RTX 5090, plus groupwise Q4 DFlash2 qualification on RTX 3090:
  `docs/results/2026-08-20-qwen38-27b-dflash2-cli.md` (this tree).

> **Retired Qwen3.8 profile:** Rows and command lines below that name
> `qwen3.8-27b/groupwise-int` or `qwen3_8_27b.ginfer` are retained only as historical measurement
> provenance. That identity is no longer registered, and those command blocks are not current run
> instructions. The current groupwise artifacts are the Q4/W8 DFlash2 companions; use the Q4
> companion with explicit `--spec none` for the corresponding non-speculative route. The historical
> values are not automatically performance claims for the replacement artifact.

The Qwen3.6 measurements characterize its three registered artifact profiles independently on one
NVIDIA GeForce RTX 5090. They cover long-context prefill and baseline decode with speculative
decoding disabled, plus long-reasoning and cross-scenario decode with MTP and DFlash. The Qwen3.6
concurrent decode-saturation campaign measures all three profiles at C=1, 2, 4, and 8. The
Qwen3.8-27B NVFP4 campaign covers the MTP0 long-context profile and the complete MTP3
speculative-decode corpus at C=1, 2, 4, and 8; its C=1 point also supplies the single-request MTP3
results below. All four current Qwen3.8-27B DFlash2 companions have a separate production-style CLI
result on RTX 5090; the groupwise Q4 companion is additionally qualified on RTX 3090. They remain
outside the multi-fixture 5090 serving campaign below.
A retired-profile PRO 6000 CLI `long_niah_8k` row is retained in the final section as historical
evidence.

The same `sm_120a` image is the product compile for NVIDIA RTX PRO 6000 Blackwell. Occupancy is
live SM count (188 on the Workstation Edition measured below). DRAM spec GB/s is the closed SKU
map in `bench/ops/ginfer_bench_common.h`: Workstation 1792 GB/s, Server 1597 GB/s. Do not label
PRO 6000 numbers as 5090. Hardware facts and Path 1/2 CLI measurements for that SKU are in the
final section; they do not replace the 5090 tables. RTX 4090 Path 1 (`sm_89`, groupwise-int,
24 GB, DRAM spec **1008 GB/s**) and RTX 3090 Path 1 (`sm_86`, groupwise-int, 24 GB, DRAM spec
**936 GB/s**) are later sections and are not a 260k campaign.

The single-request corpus requests were submitted serially to a persistent `ginfer-serve` process
over the loopback OpenAI-compatible HTTP endpoint. Each reported corpus fixture used five fixed
seeds. Values are arithmetic mean ± sample standard deviation, and server warm-up completes before
the measured requests. The concurrent campaign has its own sustained-wave method below.

## Single-request serving performance method

| Setting | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 GiB |
| CUDA compile/runtime | 13.1 / 13.1 |
| CUDA driver API | 13.3 for NVFP4 and refreshed 35B MTP3; 13.1 for the remaining single-request campaigns |
| Request mode | One active request, `stream=false` |
| Maximum context | 262,144 tokens; 131,072 for refreshed NVFP4 MTP3 |
| Prefill chunk | 1,024 tokens |
| KV cache | INT8 group-64 |
| CUDA Graph | Enabled |
| Prefix reuse | Disabled |
| Sampling | Temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.0 |
| Greedy profile | Exact argmax (`--sampling greedy` in the corpus runner) |
| MTP0 | no `--spec` |
| MTP3 | `--spec mtp --draft-tokens 3 --lm-head-draft` |
| DFlash block=8 | `--spec dflash --draft-tokens 7 --lm-head-draft` |

The MTP0 profile uses four Long NIAH prompts with approximately 8K, 64K, 128K, and 256K tokens.
Thinking is disabled and the output budget is 128 tokens. These runs measure prefill throughput,
server-internal time to first token, and baseline decode throughput at each context length. Content
scenarios are not repeated with MTP disabled because they do not change the baseline decode path.

The speculative-decode corpus contains three long-reasoning fixtures with thinking enabled and a
65,536-token output limit, followed by twelve fixtures covering code, story, translation, and
structured output. The cross-scenario fixtures disable thinking and use a 4,096-token output limit.
The tables report actual completion lengths rather than assuming that every request reaches its
limit.

Metrics are computed from the server's unrounded phase timings and speculative-decode counters:

```text
prefill_tok_s = prompt_tokens / prefill_seconds
server_ttft_ms = 1000 * (prepare_seconds + vision_seconds + prefill_seconds)
decode_tok_s = (completion_tokens - 1) / decode_seconds
spec_acceptance = accepted_tokens / drafted_tokens
spec_tokens_per_round = 1 + accepted_tokens / speculative_rounds
```

Decode throughput is a transport/execution measurement, not a correctness score. The response text,
finish reason, and fixture-level structural requirements are audited separately below. A request
that exhausts its output budget or enters a repetition loop remains useful as a sustained-decode
stress sample, but is not presented as a successfully completed task.

## Qwen3.8-27B NVFP4 concurrent MTP3 corpus makespan

This campaign uses the complete speculative-decode corpus described above: three long-reasoning
fixtures and twelve cross-scenario fixtures, each with five fixed seeds, for 75 requests. The
runner shuffles that fixed request set once with seed `20260811` and preserves the same ordered HTTP
send sequence at every concurrency. Exactly C persistent client workers each submit their next
request only after receiving the current response. C=1 is therefore a serial single-request corpus
on one persistent server and supplies the per-fixture Qwen3.8 results in the final section.

Each point starts a fresh server on an RTX 5090 with CUDA 13.1 compile/runtime, CUDA driver API
13.3, stochastic sampling, INT8 group-64 KV, a 1,024-token prefill chunk, CUDA Graphs, prefix reuse
disabled, a 131,072-token per-request context ceiling, `--kv-capacity auto`, and
`--spec mtp --draft-tokens 3 --lm-head-draft`. Makespan begins when all client workers are released
and ends when the final complete HTTP response has been read. Prefill and decode rates divide the
corresponding server token totals by that full makespan; average batch includes the entire run,
including workload transitions and drain.

| C | Requests | Computed prefill tokens | Decode tokens | Makespan (s) | Requests/s | Prefill tok/s | Decode tok/s | Avg batch | MTP acceptance | Speedup vs. C1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 75 | 15,460 | 752,160 | 4,670.27 | 0.0161 | 3.3 | 161.1 | 1.00 | 60.8% | 1.00× |
| 2 | 75 | 15,460 | 739,951 | 2,510.78 | 0.0299 | 6.2 | 294.7 | 1.98 | 59.2% | 1.86× |
| 4 | 75 | 15,460 | 713,384 | 1,647.74 | 0.0455 | 9.4 | 432.9 | 3.29 | 58.0% | 2.83× |
| 8 | 75 | 15,460 | 723,602 | 2,164.90 | 0.0346 | 7.1 | 334.2 | 2.36 | 57.6% | 2.16× |

All 300 requests completed without a request, CUDA, or out-of-memory failure. C=4 gives the
shortest complete-corpus makespan. C=8 is limited by memory pressure, which constrains effective
batching and makes the end-to-end result slower than C=4. Sampling is stochastic: prompts, seeds,
and send order are fixed, but concurrency-specific numerical routes can change sampled
continuations and their lengths. The makespan speedup is therefore a fixed-workload serving result
rather than a fixed-token normalization; the exact decode-token totals are retained in the table.

## Concurrent MTP3 decode saturation

The concurrent campaign uses the `long_decode_aime26_15` fixture with thinking enabled. The
rendered prompt is 293 tokens, and every request has an 8,192-token output budget. For each
concurrency C, the runner starts a fresh `ginfer-serve` process with `max_concurrency=C`, releases
C non-stream requests together using distinct fixed seeds, and waits for every HTTP response.
Startup and server warmup occur before the measured wave.

All points use an RTX 5090, CUDA 13.1 compile/runtime, CUDA driver API 13.3, stochastic sampling
(temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.0), INT8 group-64 KV, a 1,024-token
prefill chunk, CUDA Graphs, prefix reuse disabled, and
`--spec mtp --draft-tokens 3 --lm-head-draft`. Each request has a 16,384-token context ceiling.
`--kv-capacity auto` resolved to exactly `C * 16,384` tokens at every point.

Saturated throughput uses only complete one-second server intervals satisfying all of the following:

- computed prefill tokens are zero;
- `running=C`, `prefilling=0`, and `decode_ready=C`;
- at least one decode round completed;
- every decode round had exactly C rows.

Ramp-up, prefill, and drain intervals are excluded. The reported aggregate rate is:

```text
steady_decode_tok_s = sum(committed_decode_tokens) / sum(interval_seconds)
```

Wave makespan starts when the client threads are released and ends after the last complete HTTP
response. MTP acceptance is aggregated over the complete wave. Each row below is one sustained
wave rather than a repeated-sample mean.

| Model profile | C | Steady (s) | Avg batch | Aggregate decode tok/s | MTP acceptance | Speedup vs. C1 | Wave makespan (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-27B `groupwise-int` | 1 | 43.01 | 1.00 | 185.8 | 68.2% | 1.00× | 44.23 |
| Qwen3.6-27B `groupwise-int` | 2 | 65.01 | 2.00 | 247.0 | 69.0% | 1.33× | 66.67 |
| Qwen3.6-27B `groupwise-int` | 4 | 102.02 | 4.00 | 309.5 | 68.4% | 1.67× | 107.49 |
| Qwen3.6-27B `groupwise-int` | 8 | 118.02 | 8.00 | 535.0 | 68.3% | 2.88× | 125.20 |
| Qwen3.6-27B `nvfp4` | 1 | 39.01 | 1.00 | 202.4 | 69.3% | 1.00× | 40.46 |
| Qwen3.6-27B `nvfp4` | 2 | 39.01 | 2.00 | 399.7 | 71.4% | 1.97× | 41.82 |
| Qwen3.6-27B `nvfp4` | 4 | 44.01 | 4.00 | 699.7 | 69.3% | 3.46× | 47.92 |
| Qwen3.6-27B `nvfp4` | 8 | 55.01 | 8.00 | 1,146.9 | 68.6% | 5.67× | 58.57 |
| Qwen3.6-35B-A3B `groupwise-int` | 1 | 12.00 | 1.00 | 593.0 | 67.2% | 1.00× | 13.75 |
| Qwen3.6-35B-A3B `groupwise-int` | 2 | 17.00 | 2.00 | 877.7 | 68.2% | 1.48× | 18.87 |
| Qwen3.6-35B-A3B `groupwise-int` | 4 | 26.01 | 4.00 | 1,166.0 | 69.8% | 1.97× | 28.43 |
| Qwen3.6-35B-A3B `groupwise-int` | 8 | 48.01 | 8.00 | 1,313.8 | 67.3% | 2.22× | 50.20 |

All 45 requests reached their output limit, producing 368,640 completion tokens. The campaign
contained 608 complete full-batch steady intervals and had no request, CUDA, or out-of-memory
failure. At C=8, available device memory after startup was 2.66 GiB for 27B groupwise-int,
2.18 GiB for 27B NVFP4, and 4.38 GiB for 35B-A3B.

## Reproduction

Build `ginfer-serve` and prepare the registered `.ginfer` artifacts. The refreshed per-target
serving tables use:

```bash
python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ginfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ginfer \
  --mode mtp3 --suite corpus-makespan --concurrency 1 \
  --max-context 262144 --kv-capacity auto \
  --output profiles/bench/concurrent_corpus_35b_mtp3_20260811

python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ginfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ginfer \
  --mode mtp3 \
  --output profiles/bench/serve_corpus_27b_mtp3_20260724

python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ginfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ginfer \
  --mode mtp0 --sampling stochastic \
  --output profiles/bench/serve_corpus_27b_nvfp4_w8_20260731

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ginfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ginfer \
  --mode mtp3 --suite corpus-makespan --concurrency 1 \
  --max-context 131072 --kv-capacity auto \
  --output profiles/bench/concurrent_corpus_27b_nvfp4_mtp3_20260811

python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ginfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4.ginfer \
  --mode mtp0 --sampling stochastic \
  --output profiles/bench/serve_corpus_qwen3_8_27b_nvfp4_mtp0_20260817

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ginfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4.ginfer \
  --mode mtp3 --suite corpus-makespan \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --max-context 131072 --kv-capacity auto \
  --output profiles/bench/concurrent_corpus_qwen3_8_27b_nvfp4_mtp3_20260817
```

The concurrent decode-saturation campaigns use:

```bash
python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ginfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ginfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_27b_mtp3_20260811

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ginfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ginfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_27b_nvfp4_mtp3_20260811

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ginfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ginfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_35b_mtp3_20260811
```

Use `--mode dflash7` for the corresponding DFlash block=8 campaign; add `--sampling greedy` for
the exact-argmax profile.

Omit `--mode` and supply the two measured Qwen3.6 groupwise-int artifacts to run the complete
published Qwen3.6 MTP0/MTP3 campaign:

```bash
python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ginfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ginfer \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ginfer \
  --output profiles/bench/serve_corpus_20260720
```

For the 27B NVFP4 accuracy run, start the model service with:

```bash
build/apps/ginfer-serve out/qwen3_6_27b_nvfp4.ginfer \
  --host 127.0.0.1 --port 18080 \
  --max-context 262144 --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

Then run the repository's full 27B reasoning suite in a separate shell:

```bash
PYTHONPATH=eval eval/.venv/bin/python -m ginfer_eval run \
  --config eval/configs/qwen3_6_27b_reasoning.yaml \
  --suite reasoning_full
```

## `qwen3_6_35b_a3b`

### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 15,544.3 ± 242.4 | 500.2 ± 7.8 | 271.1 ± 3.6 |
| 64,512 | 5 | 10,809.0 ± 95.3 | 6,009.9 ± 52.6 | 242.9 ± 1.3 |
| 130,048 | 5 | 7,828.4 ± 34.1 | 16,693.3 ± 71.2 | 219.4 ± 1.6 |
| 260,096 | 5 | 5,157.1 ± 52.4 | 50,598.8 ± 519.7 | 188.2 ± 2.1 |

### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 8,223.0 ± 2,224.1 | 726.2 ± 22.9 | 82.8% ± 3.4% | 3.48 ± 0.10 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 620.3 ± 8.1 | 72.7% ± 1.4% | 3.18 ± 0.04 |
| `long_decode_aime26_30` | 5 | 52,977.8 ± 11,849.6 | 671.9 ± 8.8 | 80.1% ± 2.7% | 3.40 ± 0.08 |

### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 657.6 ± 34.3 | 70.3% ± 5.5% | 3.11 ± 0.16 |
| Story | 15 | 456.2 ± 36.6 | 38.0% ± 6.0% | 2.14 ± 0.18 |
| Translation | 15 | 649.7 ± 33.0 | 67.6% ± 5.1% | 3.03 ± 0.15 |
| Structured | 15 | 770.9 ± 29.3 | 89.1% ± 4.9% | 3.67 ± 0.15 |

### DFlash block=8 (`k=7`), stochastic sampling

The fixtures, five seeds, sampling parameters, and output limits are identical to MTP3. Different
speculative backends consume random values differently, so this is a fixed-workload comparison
rather than a token-identical paired-output comparison.

#### Long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 8,495.4 ± 2,221.2 | 764.1 ± 55.6 | 65.2% ± 5.4% | 5.56 ± 0.38 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 584.0 ± 33.3 | 51.1% ± 3.7% | 4.58 ± 0.26 |
| `long_decode_aime26_30` | 5 | 53,330.4 ± 11,198.5 | 638.3 ± 15.8 | 56.4% ± 2.5% | 4.95 ± 0.17 |

#### Cross-scenario decode

| Category | Samples | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 562.3 ± 36.2 | 43.0% ± 3.7% | 4.01 ± 0.26 |
| Story | 15 | 261.7 ± 51.1 | 12.1% ± 5.3% | 1.85 ± 0.37 |
| Translation | 15 | 490.8 ± 62.6 | 34.8% ± 6.3% | 3.44 ± 0.44 |
| Structured | 15 | 786.4 ± 124.7 | 66.5% ± 13.5% | 5.66 ± 0.94 |

#### Decode throughput versus MTP3

| Workload | MTP3 tok/s | DFlash tok/s | DFlash change |
|---|---:|---:|---:|
| `long_decode_aime26_01` | 726.2 | 764.1 | +5.2% |
| `long_decode_aime26_15` | 620.3 | 584.0 | -5.9% |
| `long_decode_aime26_30` | 671.9 | 638.3 | -5.0% |
| Code | 657.6 | 562.3 | -14.5% |
| Story | 456.2 | 261.7 | -42.6% |
| Translation | 649.7 | 490.8 | -24.5% |
| Structured | 770.9 | 786.4 | +2.0% |

### DFlash block=8 (`k=7`), greedy sampling

Greedy uses exact argmax; all other corpus and server settings remain unchanged. The five seeds
repeat the same deterministic generation path, so within-fixture standard deviation measures
runtime variation rather than output variation.

#### Long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 6,692.0 ± 0.0 | 872.4 ± 3.3 | 74.4% ± 0.0% | 6.21 ± 0.00 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 651.6 ± 0.6 | 58.6% ± 0.0% | 5.10 ± 0.00 |
| `long_decode_aime26_30` | 5 | 65,536.0 ± 0.0 | 994.9 ± 3.4 † | 98.0% ± 0.0% | 7.86 ± 0.00 |

† The generation is a deterministic repetition loop, not a valid AIME response. The raw rate is
retained to describe what was measured, but is excluded from performance comparisons.

#### Cross-scenario decode

| Category | Samples | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 599.8 ± 12.3 | 46.4% ± 1.4% | 4.25 ± 0.10 |
| Story | 15 | 291.5 ± 55.6 | 14.9% ± 5.7% | 2.04 ± 0.40 |
| Translation | 15 | 475.5 ± 50.6 | 33.0% ± 5.1% | 3.31 ± 0.36 |
| Structured | 15 | 869.0 ± 120.2 | 74.5% ± 13.1% | 6.21 ± 0.92 |

#### Decode throughput versus stochastic DFlash

| Workload | Stochastic tok/s | Greedy tok/s | Greedy change |
|---|---:|---:|---:|
| `long_decode_aime26_01` | 764.1 | 872.4 | +14.2% |
| `long_decode_aime26_15` | 584.0 | 651.6 | +11.6% |
| `long_decode_aime26_30` | 638.3 | 994.9 † | not comparable † |
| Code | 562.3 | 599.8 | +6.7% |
| Story | 261.7 | 291.5 | +11.4% |
| Translation | 490.8 | 475.5 | -3.1% |
| Structured | 786.4 | 869.0 | +10.5% |

### Speculative-decode output audit

The audit covers all 225 stored July responses from the 35B-A3B MTP3 stochastic-sampler, DFlash
stochastic-sampler, and DFlash greedy campaigns. It checks termination, exact repetition, and
fixture-specific mechanical constraints. AIME 1 was checked algebraically; the AIME 30 answer
(`393`) was checked by independent enumeration. This audit does not attempt to assign a subjective
quality score to prose or translations.

#### Long-reasoning answers

| Fixture | MTP3 stochastic sampler | DFlash stochastic sampler | DFlash greedy |
|---|---|---|---|
| `long_decode_aime26_01` | 5/5 correct, natural stop | 5/5 correct, natural stop | 5/5 correct, natural stop |
| `long_decode_aime26_15` | 0/5 answers; all reach 65,536-token limit | 0/5 answers; all reach 65,536-token limit | 0/5 answers; all reach 65,536-token limit |
| `long_decode_aime26_30` | 3/5 correct, 1 wrong, 1 no answer | 2/5 correct, 1 wrong, 2 no answer | 0/5 answers; all enter the same repetition loop |

The greedy AIME 30 response has an empty final-content field and fills its 65,536-token reasoning
budget. The exact line `Wait, $x_7 x_1 x_3$ is $x_7 x_1 x_3$.` occurs 2,406 times among 2,538
non-empty reasoning lines. Its 98.0% acceptance and 994.9 tok/s therefore characterize a highly
predictable pathological loop, not normal reasoning performance.

AIME 15 is also not a valid completion in any of the three campaigns: every sample exhausts the
budget without a boxed answer. Its output is long, non-convergent reasoning rather than the short
exact cycle seen in greedy AIME 30. The AIME 15 rates may be read only as sustained long-decode
throughput.

#### Cross-scenario outputs

| Category | MTP3 stochastic sampler | DFlash stochastic sampler | DFlash greedy |
|---|---|---|---|
| Code | 1/15 natural stops; 0/15 prompt-complete | 2/15 natural stops; 0/15 prompt-complete | 0/15 natural stops |
| Story | 9/15 natural stops; the nine Chinese outputs pass requested division and minimum length | 8/15 natural stops; the eight Chinese outputs pass requested division and minimum length | 10/15 natural stops; five Chinese dialogue outputs are under length |
| Translation | 15/15 natural stops; 15/15 pass structural checks | 15/15 natural stops; 15/15 pass structural checks | 15/15 natural stops; 15/15 pass structural checks |
| Structured | 0/15 satisfy the requested complete record/script contract | 0/15 satisfy the requested complete record/script contract | 0/15 satisfy the requested complete record/script contract |

The code prompts require complete runnable multi-file deliverables, but almost all outputs end at the
4,096-token limit. The three natural-stop exceptions also contain decisive contract failures: the
MTP3 CUDA response substitutes CUDA 12.8 and an older architecture list; the DFlash CUDA response
copies FP32 input into a half-sized 16-bit allocation and passes raw `unsigned short` values to BF16
intrinsics; and the DFlash Python response never writes its advertised JSONL event stream to the
configured log file. Code throughput is therefore a truncated-generation stress result, not
successful code-generation throughput.

All English mystery samples reach the output limit with an unfinished ending. The naturally stopped
Chinese stories have the requested chapter/act counts; the MTP3 and stochastic-DFlash samples also
meet their requested Chinese-character minima. Greedy's five dialogue stories contain 3,239 Chinese
characters each, below the requested 3,500. Story results are consequently a mixed normal/truncated
workload.

All translation outputs stop naturally. Each plain-document result preserves six sections and
provides at least twenty glossary entries; each Markdown result preserves heading levels, the
six-line table, all required inline identifiers, and the exact fenced JSON object. Translation is
the cleanest cross-scenario normal-completion comparison in this corpus.

The structured prompts intentionally exceed what these generations fit into 4,096 tokens. MTP3,
stochastic DFlash, and greedy DFlash produce only 49–60, 49–58, and 57 valid JSONL records,
respectively, versus the requested 160. Their complete-width CSV ranges are 122–139, 121–143, and
133 rows versus the requested 220. No SQL output satisfies all four tables, two views, at least 80
rows, and six final analytical queries. These high-acceptance results describe predictable partial
record generation only.

The exact-line and repeated-token scan found no other response with a short-cycle collapse comparable
to greedy AIME 30. Output-limit and prompt-compliance failures above remain material even when no
repetition loop is present.

## `qwen3_6_27b`

### EvalScope reasoning accuracy

Both weight profiles were evaluated through GInfer's OpenAI-compatible serving route with thinking
enabled, MTP=3, and a 262,144-token context limit. EvalScope 1.9.0 used 0-shot prompts, rule-based
scoring, and one sample per problem with temperature 0.6, top-p 0.95, top-k 20, presence penalty
1.0, and seed 42. All 258 samples completed and were scored for each profile.

| Weights ID | AIME 2025 | AIME 2026 | GPQA-Diamond |
|---|---:|---:|---:|
| `groupwise-int` | 86.67% (26 / 30) | 93.33% (28 / 30) | 86.87% (172 / 198) |
| `nvfp4` | 93.33% (28 / 30) | 93.33% (28 / 30) | 84.34% (167 / 198) |

These are single-sample results under the stated evaluation profile, not pass@k scores. Each
benchmark remains independently reportable; no combined score is computed.

### `groupwise-int`

#### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 3,218.1 ± 4.3 | 2,392.4 ± 3.0 | 77.6 ± 0.1 |
| 64,512 | 5 | 2,655.9 ± 2.9 | 24,335.7 ± 25.2 | 70.7 ± 0.1 |
| 130,048 | 5 | 2,185.3 ± 0.3 | 59,590.3 ± 8.9 | 64.5 ± 0.1 |
| 260,096 | 5 | 1,614.8 ± 0.6 | 161,221.8 ± 62.5 | 54.8 ± 0.1 |

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 10,686.2 ± 553.8 | 175.4 ± 1.0 | 77.9% ± 0.9% | 3.34 ± 0.03 |
| `long_decode_aime26_15` | 5 | 61,604.2 ± 5,677.9 | 161.9 ± 2.8 | 73.4% ± 1.7% | 3.20 ± 0.05 |
| `long_decode_aime26_30` | 5 | 47,339.8 ± 9,162.2 | 172.2 ± 0.9 | 78.8% ± 0.8% | 3.36 ± 0.02 |

#### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 167.0 ± 5.4 | 72.3% ± 3.5% | 3.17 ± 0.11 |
| Story | 15 | 112.6 ± 9.4 | 37.8% ± 5.9% | 2.13 ± 0.18 |
| Translation | 15 | 161.5 ± 11.3 | 68.3% ± 7.2% | 3.05 ± 0.22 |
| Structured | 15 | 193.0 ± 18.8 | 88.7% ± 11.7% | 3.66 ± 0.35 |

### `nvfp4`

The fixtures, seeds, sampling parameters, output limits, and runtime options are identical to the
groupwise-int serving campaign. Quantization can change sampled tokens, so the MTP3 results are a
fixed-workload comparison rather than a token-identical output comparison.

#### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 11,191.5 ± 70.2 | 692.5 ± 4.3 | 86.4 ± 0.5 |
| 64,512 | 5 | 6,298.5 ± 97.6 | 10,288.6 ± 159.3 | 78.0 ± 1.2 |
| 130,048 | 5 | 4,204.7 ± 14.1 | 31,012.5 ± 104.6 | 71.2 ± 0.2 |
| 260,096 | 5 | 2,510.6 ± 16.8 | 103,761.1 ± 698.8 | 59.9 ± 0.3 |

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 12,053.4 ± 820.9 | 231.0 ± 3.0 | 80.2% ± 1.2% | 3.41 ± 0.04 |
| `long_decode_aime26_15` | 5 | 63,109.0 ± 5,426.9 | 213.1 ± 4.2 | 76.3% ± 2.0% | 3.29 ± 0.06 |
| `long_decode_aime26_30` | 5 | 57,166.4 ± 9,204.9 | 223.3 ± 1.8 | 81.1% ± 1.5% | 3.43 ± 0.04 |

#### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 220.3 ± 8.2 | 74.2% ± 4.0% | 3.23 ± 0.12 |
| Story | 15 | 148.8 ± 11.6 | 39.2% ± 5.7% | 2.18 ± 0.17 |
| Translation | 15 | 213.6 ± 12.2 | 70.5% ± 6.0% | 3.12 ± 0.18 |
| Structured | 15 | 252.2 ± 16.3 | 89.8% ± 8.0% | 3.69 ± 0.24 |

The baseline and speculative-decode suites intentionally measure different supported workloads.
No per-scenario baseline/speculative speedup is reported.

## `qwen3_8_27b`

### `nvfp4`

The MTP0 table comes from the serial Long NIAH campaign described by the single-request method. The
MTP3 tables come from the C=1 point of the fixed concurrent-corpus campaign, which serially runs the
same three long-reasoning and twelve cross-scenario fixtures. Each fixture has five fixed seeds. The
tables report arithmetic mean ± sample standard deviation from the server's per-request phase
timings and speculative counters.

#### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 8,340.4 ± 13.0 | 931.6 ± 1.6 | 71.2 ± 0.1 |
| 64,512 | 5 | 5,297.9 ± 259.2 | 12,281.1 ± 561.5 | 65.7 ± 0.8 |
| 130,048 | 5 | 3,544.7 ± 25.3 | 36,853.5 ± 259.4 | 59.6 ± 0.9 |
| 260,096 | 5 | 2,203.1 ± 13.4 | 118,354.8 ± 717.2 | 52.9 ± 2.3 |

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 1,465.4 ± 417.3 | 195.2 ± 4.6 | 76.0% ± 2.4% | 3.28 ± 0.07 |
| `long_decode_aime26_15` | 5 | 65,414.4 ± 271.9 | 151.4 ± 2.0 | 56.2% ± 1.1% | 2.69 ± 0.03 |
| `long_decode_aime26_30` | 5 | 50,023.4 ± 14,839.1 | 167.5 ± 23.7 | 64.6% ± 14.9% | 2.94 ± 0.45 |

#### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 194.3 ± 6.1 | 76.4% ± 3.9% | 3.29 ± 0.12 |
| Story | 15 | 126.1 ± 10.9 | 37.4% ± 5.8% | 2.12 ± 0.17 |
| Translation | 15 | 192.3 ± 11.9 | 75.0% ± 6.5% | 3.25 ± 0.19 |
| Structured | 15 | 219.8 ± 8.6 | 90.8% ± 5.1% | 3.72 ± 0.15 |

## RTX PRO 6000 Blackwell Workstation Edition

This section records the S1 `sm_120a` image on one NVIDIA RTX PRO 6000 Blackwell **Workstation
Edition**. It does not replace the 5090 tables above. The name does not contain `Server`; the closed
bench DRAM map therefore uses **1792 GB/s**. The Server edition of the same product family maps to
**1597 GB/s**. Do not label these numbers as 5090.

| Field | Value |
|---|---|
| Host | `AIS-2-8592-L01` |
| GPU name (`DeviceContext` / `nvidia-smi`) | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| Product brand | NVIDIA RTX |
| Edition | Workstation (not Server) |
| Compute capability | 12.0 |
| SM count | 188 |
| Occupancy (live SM) | `sm_arch=Blackwell120`, `gqa_decode_splits_base=94` (not 5090's 85) |
| GDN cooperative residency | 27B Split8/4/2: 376 CTAs; 35B Split32: 376 CTAs; 35B Split16/8/4/2: 752 CTAs |
| VRAM (`nvidia-smi`) | 97887 MiB |
| VRAM (`DeviceContext::total_vram`) | 101964644352 bytes (97241.1 MiB) |
| DRAM spec (SKU map) | 1792 GB/s Workstation; 1597 GB/s Server |
| CUDA compile | 13.3.73 (`/usr/local/cuda/bin/nvcc`; toolkit floor 13.1) |
| Driver | 595.71.05 (CUDA driver API 13.2) |
| Image | `CMAKE_CUDA_ARCHITECTURES=120a` Release; `cuobjdump --list-elf` includes `nvfp4_w4a4_tma.sm_120a.cubin` and `nvfp4_linear_swiglu_w4a4_tma.sm_120a.cubin` |

The live device policy on this host is `sm_count=188`, `cc=120`, `gqa_splits=94`, `tma=1`,
`nvfp4=1`. GDN residency is owned by each compiled kernel rather than generic two/four-CTA
device multipliers. `ginfer_device_test` exits 0.

The retired Path 1 (`qwen3_8_27b.ginfer`, `groupwise-int`) and current Path 2
(`qwen3_8_27b_nvfp4.ginfer`) CLI both
start with `--kv-capacity auto`. Short text generate (`--prompt "Return one sentence." --max-new 64
--no-thinking`) succeeded at MTP 0 and MTP 3 (`--spec mtp --draft-tokens 3 --lm-head-draft`). Auto
KV on the default 2,048-token context resolved to 2,048 tokens; free-after-weights was 77.62 GiB
(groupwise-int) and 74.56 GiB (NVFP4). The 96 GiB SKU does not trip 5090 leftover arithmetic:
`--kv-capacity auto` uses live `cudaMemGetInfo` after weights.

A 7,680-token `long_niah_8k` prefill (T≥1024, `--prefill-chunk 1024`) launched for both identities,
including NVFP4 W4A4/TMA and GDN cooperative grids at the 188-SM per-kernel resident caps (376 for
27B and 376/752 by 35B schedule). No CUDA launch failure. Linear T catalogs were not reswept:
NVFP4 7,680-token prefill is
faster than the 5090 Qwen3.8-27B NVFP4 row (11,027 vs 8,340 tok/s), and decode is 72.1 vs 71.2
tok/s. That is not an occupancy hole versus 188 vs 170 SMs.

### `long_niah_8k` CLI subset (C=1)

CLI equivalent of the published MTP0 7,680-token row, not a persistent `ginfer-serve` HTTP campaign
and not a 260k sweep. Each identity used the five corpus seeds, INT8 group-64 KV, CUDA Graphs, a
1,024-token prefill chunk, `--max-context 262144 --kv-capacity auto` (resolved to 262,144 tokens),
thinking disabled, and sampling temperature 0.6 / top-p 0.95 / top-k 20 / presence penalty 1.0.
Values are arithmetic mean ± sample standard deviation from the CLI phase timings. CLI TTFT is
`1000 * (prepare_seconds + prefill_seconds)`. Every sample produced the fixture oracle
`ORCHID=493817; COLOR=COBALT` and stopped on a stop-token.

| Identity | Samples | Prefill tok/s | CLI TTFT (ms) | Decode tok/s |
|---|---:|---:|---:|---:|
| Qwen3.8-27B `groupwise-int` | 5 | 3,929.7 ± 19.8 | 1,974.0 ± 10.5 | 80.3 ± 1.5 |
| Qwen3.8-27B `nvfp4` | 5 | 11,027.1 ± 36.2 | 715.8 ± 2.6 | 72.1 ± 0.1 |

```bash
./build/apps/ginfer models/qwen3_8_27b.ginfer \
  --messages examples/cli/messages/long_niah_8k.json \
  --max-context 262144 --kv-capacity auto --kv-dtype int8 --prefill-chunk 1024 \
  --max-new 128 --no-thinking \
  --temperature 0.6 --top-p 0.95 --top-k 20 --presence-penalty 1.0 \
  --seed 7632647173703958409

./build/apps/ginfer models/qwen3_8_27b_nvfp4.ginfer \
  --messages examples/cli/messages/long_niah_8k.json \
  --max-context 262144 --kv-capacity auto --kv-dtype int8 --prefill-chunk 1024 \
  --max-new 128 --no-thinking \
  --temperature 0.6 --top-p 0.95 --top-k 20 --presence-penalty 1.0 \
  --seed 7632647173703958409
```

## RTX 3090 (`sm_86`)

This section records the S2 `sm_86` image on one NVIDIA GeForce RTX 3090. It does not replace the
5090 tables or the PRO 6000 section above. Its Path 1 data is historical evidence from the retired
companion-free groupwise identity. There is no FP8 and
no NVFP4: Engine load of `qwen3.8-27b/nvfp4` throws `nvfp4 is not supported on this GInfer image`.
The closed bench DRAM map uses **936 GB/s**. L2 is 6 MiB; Linear T catalogs were not retuned.
These numbers are a short-context smoke, not a 5090 260k campaign.

| Field | Value |
|---|---|
| Host | `AIS-1-2950X-L02` |
| GPU name (`DeviceContext` / `nvidia-smi`) | NVIDIA GeForce RTX 3090 |
| Compute capability | 8.6 |
| SM count | 82 |
| Occupancy (live SM) | `sm_arch=Ampere86`, `gqa_decode_splits_base=41` |
| 27B GDN cooperative residency | Split8: 164 CTAs; Split4/2: 82 CTAs. At T=1024 the 192-CTA Split8 and 96-CTA Split4 grids do not fit, so the resolver selects the 48-CTA Split2 schedule |
| 35B GDN cooperative residency | Split32: 164 CTAs; Split16: 328 CTAs; Split8/4/2: 246 CTAs. At T=1024 the 256-CTA Split8 grid does not fit, so the resolver selects the 128-CTA Split4 schedule |
| VRAM (`nvidia-smi`) | 24576 MiB |
| VRAM (`DeviceContext::total_vram`) | 25288507392 bytes (24117.0 MiB) |
| DRAM spec (SKU map) | 936 GB/s |
| CUDA compile | 13.1.115 (`/usr/local/cuda/bin/nvcc`; toolkit floor 13.1) |
| Driver | 580.159.03 (nvidia-smi CUDA 13.0) |
| Image | `CMAKE_CUDA_ARCHITECTURES=86` Release; `cuobjdump --list-elf` is `sm_86` only; no `nvfp4` ELF; `ginfer_nvfp4_tma` omitted |

The live device policy on this host is `sm_count=82`, `cc=86`, `gqa_splits=41`, `tma=0`,
`nvfp4=0`, `pdl=0`. The compiled-kernel resource table above replaces the old generic GDN
two/four-CTA multipliers. `ginfer_device_test` exits 0.

### 24 GB envelope

Qwen3.8-27B `groupwise-int` weights occupy 15.92 GiB. Free after weights is **7.19 GiB**
(7,724,926,976 bytes). Automatic KV keeps 1 GiB headroom. `--kv-capacity auto` cannot exceed
`--max-context` at C=1, so the default CLI (`--max-context 2048`, BF16 KV) admits **2048** tokens.

Largest Engine start with no Vision, no speculative backend, `--prefill-chunk 1024`, and 1 GiB
auto headroom:

| KV dtype | Largest `--max-context` that starts | Next 64-token step |
|---|---:|---|
| BF16 | 93,568 | 93,696 fails |
| INT8 group-64 | 181,504 | 181,760 fails |

`--max-context 262144` does **not** start (INT8 minimum reservation 9.37 GiB plus 1 GiB headroom
exceeds the 7.19 GiB leftover). The 24 GB product context is
`--max-context 131072 --kv-dtype int8 --kv-capacity auto` (under the 181,504 INT8 start limit).

### CLI smoke

Path 1 (`qwen3_8_27b.ginfer`, `groupwise-int`) short text generate (`--prompt "Return one
sentence." --max-new 64 --no-thinking --kv-capacity auto`) succeeded. Auto KV on the default
2,048-token BF16 context resolved to 2,048 tokens. Nine tokens, stop-token, prefill 132.48 tok/s,
decode 40.18 tok/s.

A 7,680-token `long_niah_8k` prefill (`--prefill-chunk 1024`, T≥1024) launched after the 27B
cooperative candidate honored the per-kernel Ampere residency (the T=1024 route is Split2, not the
192-CTA Split8). Oracle
`ORCHID=493817; COLOR=COBALT`. Single seed, not a five-seed campaign.

| Identity | Samples | Prefill tok/s | Decode tok/s | Auto KV |
|---|---:|---:|---:|---|
| Qwen3.8-27B `groupwise-int` | 1 | 972.93 | 38.90 | 8,192 INT8 |

```bash
cmake -S . -B build-86 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-86 --parallel --target ginfer

./build-86/apps/ginfer models/qwen3_8_27b.ginfer \
  --prompt "Return one sentence." --max-new 64 --kv-capacity auto --no-thinking

./build-86/apps/ginfer models/qwen3_8_27b.ginfer \
  --messages examples/cli/messages/long_niah_8k.json \
  --max-context 8192 --kv-capacity auto --kv-dtype int8 --prefill-chunk 1024 \
  --max-new 128 --no-thinking \
  --temperature 0.6 --top-p 0.95 --top-k 20 --presence-penalty 1.0 \
  --seed 7632647173703958409
```

## RTX 4090 (`sm_89`)

This section records the WSL `sm_89` image on one NVIDIA GeForce RTX 4090. It does not replace the
5090 tables, the PRO 6000 section, or the 3090 section. Its Path 1 data is historical evidence from
the retired companion-free groupwise identity. There is
no FP8 and no NVFP4: Engine load of `qwen3.8-27b/nvfp4` throws
`nvfp4 is not supported on this GInfer image`. The closed bench DRAM map uses **1008 GB/s**.
Linear T catalogs were not retuned. These numbers are a short-context smoke, **not** comparable to
the 5090 260k campaign.

| Field | Value |
|---|---|
| Host | `DESKTOP-DG588BU` (WSL2) |
| GPU name (`DeviceContext` / `nvidia-smi`) | NVIDIA GeForce RTX 4090 |
| Compute capability | 8.9 |
| SM count | 128 |
| Occupancy (live SM) | `sm_arch=Ada89`, `gqa_decode_splits_base=64` |
| 27B GDN cooperative residency | Split8: 256 CTAs; Split4/2: 128 CTAs. The 192-CTA catalog Split8 grid at T=1024 fits |
| 35B GDN cooperative residency | Split32: 256 CTAs; Split16: 512 CTAs; Split8/4/2: 384 CTAs |
| VRAM (`nvidia-smi`) | 24564 MiB (616 MiB used at smoke; no compute app) |
| VRAM (`DeviceContext::total_vram`) | 25756696576 bytes (24564.0 MiB) |
| DRAM spec (SKU map) | 1008 GB/s |
| CUDA compile | 13.3.73 (`nvidia/cu13` `nvcc`; toolkit floor 13.1) |
| Driver | 596.21 (nvidia-smi CUDA 13.2) |
| Image | `CMAKE_CUDA_ARCHITECTURES=89` Release; `cuobjdump --list-elf` is `sm_89` only; no `nvfp4` ELF; `ginfer_nvfp4_tma` omitted |

The live device policy on this host is `sm_count=128`, `cc=89`, `gqa_splits=64`, `tma=0`,
`nvfp4=0`, `pdl=0`. The compiled-kernel resource table above replaces the old generic GDN
two/four-CTA multipliers. `ginfer_device_test` exits 0.

### 24 GB envelope

Qwen3.8-27B `groupwise-int` weights occupy 15.92 GiB. Free after weights is **6.53 GiB**
(7,011,174,400 bytes). Automatic KV keeps 1 GiB headroom. `--kv-capacity auto` cannot exceed
`--max-context` at C=1, so the default CLI (`--max-context 2048`, BF16 KV) admits **2048** tokens.

Largest Engine start with no Vision, no speculative backend, `--prefill-chunk 1024`, and 1 GiB
auto headroom:

| KV dtype | Largest `--max-context` that starts | Next 64-token step |
|---|---:|---|
| BF16 | 82,688 | 82,752 fails |
| INT8 group-64 | 160,448 | 160,512 fails |

`--max-context 262144` does **not** start (INT8 minimum reservation 8.73 GiB plus 1 GiB headroom
exceeds the 6.53 GiB leftover). The 24 GB product context is
`--max-context 131072 --kv-dtype int8 --kv-capacity auto` (verified Engine start: KV **131,072**,
INT8 payload 4.12 GiB, 1.93 GiB free after startup).

### CLI smoke

Path 1 (`qwen3_8_27b.ginfer`, `groupwise-int`) short text generate (`--prompt "Return one
sentence." --max-new 64 --no-thinking --kv-capacity auto`) succeeded. Auto KV on the default
2,048-token BF16 context resolved to 2,048 tokens. Nine tokens, stop-token, prefill 251.38 tok/s,
decode 49.33 tok/s. PDL is off (`GINFER_HAS_PDL=0`); Q4 GEMV decode did not raise
`cudaErrorInvalidValue`.

The recorded 7,680-token `long_niah_8k` prefill (`--prefill-chunk 1024`, T≥1024) predates the exact
per-kernel residency policy and used the former conservative one-CTA/SM rule (no 192-CTA Split8).
The current SM89 resolver admits Split8 at T=1024; this historical performance row has not been
remeasured under that faster route. Oracle `ORCHID=493817; COLOR=COBALT`. Single seed, not a
five-seed campaign.

| Identity | Samples | Prefill tok/s | Decode tok/s | Auto KV |
|---|---:|---:|---:|---|
| Qwen3.8-27B `groupwise-int` | 1 | 2246.66 | 48.96 | 8,192 INT8 |

### 1k / 1k CLI (131k INT8 pool)

Qwen3.8-27B `groupwise-int`, `--max-context 131072 --kv-dtype int8 --kv-capacity auto
--prefill-chunk 1024 --no-thinking --greedy`. Prompt **999** tokens, generate **1000** tokens
(`finish reason` `output-limit`). Single run, not a five-seed campaign. KV pool **131,072**.

| Identity | Prompt tok | Gen tok | Prefill | Decode | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B `groupwise-int` | 999 | 1000 | 0.469 s | 20.033 s | 2129.70 | 49.87 |

```bash
cmake -S . -B build-89 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-89 --parallel --target ginfer

./build-89/apps/ginfer models/qwen3_8_27b.ginfer \
  --prompt "Return one sentence." --max-new 64 --kv-capacity auto --no-thinking

./build-89/apps/ginfer models/qwen3_8_27b.ginfer \
  --messages /path/to/999-token-prompt.json \
  --max-context 131072 --kv-capacity auto --kv-dtype int8 --prefill-chunk 1024 \
  --max-new 1000 --no-thinking --greedy
```

### CLI vs llama.cpp (MTP3 instruct)

Same 999/1000 CLI, thinking off, MTP window 3, Unsloth instruct sampling (temp 0.7 / top-p 0.80 /
top-k 20 / presence 1.5). GInfer `groupwise-int` INT8 KV versus ggml-org llama.cpp Q4_K_M GGUF
`q8_0` KV. Full protocol and later-host rows:
[results/2026-08-19-qwen38-27b-cli-vs-llamacpp.md](results/2026-08-19-qwen38-27b-cli-vs-llamacpp.md).

| Engine | Prefill tok/s | Decode tok/s | MTP accept |
|---|---:|---:|---|
| GInfer sm_89 | 2048.07 | 106.65 | 49.92% (599/1200), 2.50 tok/round |
| llama.cpp b10523 `d59d455fd` | 1047.00 | 75.25 | 53.84% (617/1146), mean len 2.62 |

llama.cpp has no `--mtp`; the flags are `--spec-type draft-mtp --spec-draft-n-max 3`. CUDA 12.8
`nvcc` ICE’d this tree; the measured binary is CUDA 13.3.
