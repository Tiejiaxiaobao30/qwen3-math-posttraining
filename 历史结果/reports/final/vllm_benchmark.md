# Final vLLM serving benchmark

## Scope and model composition

Serving-only closeout of `Qwen/Qwen3-1.7B-Base` + **SFT seed42 LoRA**, merged with `merge_and_unload()` and served as `qwen3-math-sft42`. The recorded tokenizer/model directory is `/kaggle/working/models/Qwen3-1.7B-SFT42-Merged`. This benchmark does not run GRPO or evaluate mathematical correctness. No training, post-test tuning, new test generation, or change to the [sealed official-test conclusions](official_test_report.md) was performed during this closeout.

The directory/name records the reported model composition; the serving archive does not contain a merged-weight hash that independently proves its identity. Weights and the serving environment are not Git artifacts.

## Environment and provenance

| Item | Recorded value | Evidence |
|---|---|---|
| Platform | Kaggle | Historical model paths and logs |
| Python | 3.12.15 | User-provided run environment; no version snapshot in ZIP |
| vLLM | 0.31.0 | Engine startup in `server.log` |
| PyTorch | 2.13.0+cu132 | User-provided run environment; no version snapshot in ZIP |
| CUDA runtime | 13.2 | User-provided run environment; no version snapshot in ZIP |
| Serving GPU | 1×Tesla T4, 15360 MiB | T4 inventory; single-device engine with TP=1, PP=1, DP=1 |
| NVIDIA driver | 580.178.04 | `hardware.txt` |
| Precision / quantization | FP16 (`torch.float16`), no quantization | `server.log` |
| Tensor parallelism | 1 | `server.log` |
| Maximum model length | 2048 | `server.log` |
| GPU memory utilization | 0.85 | `server.log` |
| Endpoint | `http://127.0.0.1:8000/v1/completions`, streaming OpenAI backend | Benchmark argument logs |

The hardware inventory has **two T4 rows**; it describes GPUs visible in the host, not two GPUs used for inference. The server log records a single-device engine. The exact physical GPU index is not preserved. Python/PyTorch/CUDA versions above are reported metadata, not an independently verified package lock. Serving FP16 is separate from the frozen BF16 official evaluation.

## Fixed workload and recorded arguments

- 64 measured synthetic random requests per run, each with exactly 256 input and 128 output tokens (`random_range_ratio=0`). JSON totals are 16384 input and 8192 output tokens for each run.
- 4 warmup requests per run (`num_warmups=4`), excluded from the 64 measured requests; both logs record completed warmup.
- `ignore_eos=True`, `request_rate=inf`, streaming enabled, no beam search. C1/C8 denote configured `max_concurrency=1/8`.
- C1 random seed **101**; C8 random seed **202**. Token lengths/load settings are fixed, but prompt contents are not paired. This is not a byte-identical fixed-prompt comparison.
- The raw JSON also contains derived `max_concurrent_requests` values (2/16). These are retained as emitted by the tool; they are not the configured client limits used to label C1/C8.

## Results

Values below are rounded from the unmodified raw JSON. C1 retains the requested finer precision; C8 follows the console's two-decimal presentation.

| Metric | C1 | C8 |
|---|---:|---:|
| Completed / requested | 64 / 64 | 64 / 64 |
| Failed | 0 | 0 |
| Measured duration (s) | 163.4411 | 29.97 |
| Request throughput (req/s) | 0.391578 | 2.14 |
| Output throughput (tok/s) | 50.1220 | 273.31 |
| Peak output throughput (tok/s, tool-reported) | 59 | 314 |
| Total throughput, input + output (tok/s) | 150.3661 | 819.93 |
| Mean TTFT (ms) | 70.372 | 322.78 |
| P99 TTFT (ms) | 76.015 | 437.44 |
| Mean TPOT (ms/token) | 19.551 | 26.94 |
| P99 TPOT (ms/token) | 22.795 | 29.12 |
| Mean E2E latency (ms) | 2553.312 | 3743.99 |
| P99 E2E latency (ms) | 2969.379 | 3809.44 |

Throughput is an **aggregate server metric**, computed across all measured requests over measured wall time: requests/s = completed / duration, output tok/s = total output tokens / duration, and total tok/s = (input + output tokens) / duration. It is not a per-user decoding speed. Peak output throughput is the tool's `max_output_tokens_per_s`, not sustained throughput.

TTFT is time to first token, TPOT is time per output token after the first token, and E2E is request completion latency. These are the tool-reported request timings; client-side admission queue time is stored separately in `queue_times`. They should not be read as the total time a request waits from the start of an infinite-rate batch.

C8 records about **5.45×** the aggregate output throughput of C1, alongside higher mean TTFT (70.372→322.78 ms) and E2E latency (2553.312→3743.99 ms). There is only one run per condition, different random prompts, 64 samples, and no confidence interval; this describes these two runs and does not establish causality, an optimal concurrency, production capacity, or quality-preserving acceleration. P99 estimates on 64 requests are especially limited. The server log also records a Triton JIT compilation warning before these benchmark timestamps; no cold-start guarantee is claimed.

## Commands reconstructed from the saved arguments

These commands document the historical configuration and were **not executed during closeout**. Activate a separate compatible serving environment and reconstruct the merged SFT42 model outside Git before any future reproduction. Use a new output directory to avoid overwriting the archived evidence.

```bash
vllm serve /kaggle/working/models/Qwen3-1.7B-SFT42-Merged \
  --served-model-name qwen3-math-sft42 --host 127.0.0.1 \
  --dtype half --tensor-parallel-size 1 --max-model-len 2048 \
  --gpu-memory-utilization 0.85
```

```bash
# C1; for C8 change --seed to 202, --max-concurrency to 8,
# and --result-filename to sft42_t4_c8.json.
vllm bench serve --backend openai --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions --model qwen3-math-sft42 \
  --tokenizer /kaggle/working/models/Qwen3-1.7B-SFT42-Merged \
  --dataset-name random --seed 101 --num-prompts 64 \
  --random-input-len 256 --random-output-len 128 --random-range-ratio 0 \
  --num-warmups 4 --ignore-eos --request-rate inf --max-concurrency 1 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --save-result --save-detailed --metadata output_len=128 \
  --result-dir /kaggle/working/vllm_benchmark_reproduction_new \
  --result-filename sft42_t4_c1.json
```

## Preserved evidence

Source: local downloaded `vllm_benchmark_final.zip` (201459 bytes), SHA256 `eec9ebd21ad39e531a514e3f6c03ab85c4a3bc6b6a1bd5780bb5ce0fd59e5b03`. The following files were selectively extracted and verified byte-for-byte against archive members:

- [C1 raw JSON](../../results/vllm/sft42_t4_c1.json) and [argument/result log](../../results/vllm/sft42_t4_c1.log).
- [C8 raw JSON](../../results/vllm/sft42_t4_c8.json) and [argument/result log](../../results/vllm/sft42_t4_c8.log).
- [Hardware inventory](../../results/vllm/hardware.txt) and [server startup/runtime log](../../results/vllm/server.log).

Raw JSON includes per-request timings and generated text from synthetic prompts. Generated text is benchmark output, not instruction or correctness evidence. Original paths and timestamps are retained as historical metadata. No `server.pid`, merged model, `vllm-env`, archive, or large weights are included in this change. The frozen official-test files and conclusions are unchanged.
