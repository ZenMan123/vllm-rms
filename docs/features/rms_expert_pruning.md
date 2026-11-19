# RMS expert pruning for routed MoE models

RMS expert pruning uses a calibrated table to omit low-score expert assignments
before the expert matrix multiplications. It is opt-in through the HF
configuration field `rms_expert_pruning`. Models using the decomposed FusedMoE execution path can prune the final
assignments returned by their native router.

The implementation uses unquantized experts on CUDA. Granite retains its fused
top-k implementation. Gemma 4 MoE uses its native router followed by RMS. Backend selection uses the normal vLLM oracle, restricted to kernels
that declare support for pruning individual expert assignments. You can omit
`--moe-backend` to use `auto`, or select Triton explicitly:

| Backend | Device / dtype | Local validation |
| --- | --- | --- |
| `triton` | CUDA, BF16/FP16 | A100 routing, expert execution and LLM tests |
| `auto` | Selects Triton for RMS | A100 routing, expert execution and LLM tests |

Only Triton declares pruning support in this change. An explicitly requested
incompatible backend fails during initialization. FlashInfer adapters are a
separate follow-up and are not included here. Without RMS, `auto` retains the
original backend selection and priorities.

Tensor and pipeline parallelism use the original global expert and decoder-layer indices. Expert
parallelism, EPLB, LoRA, routing simulation and expert substitution are rejected
during initialization. Multi-GPU configurations have not yet been validated.
The checks below validate CUDA routing and expert dispatch. End-to-end speedup
depends on the workload and must be measured with the target batch shapes.

## Export a calibration table

The [research repository](https://github.com/paiksonn/MoE-Inference-Acceleration)
contains the original calibration bundles and the reference routing policies.
From a checkout with this vLLM installed, export a selected variant:

```bash
python examples/offline_inference/moe/export_rms_tables.py \
  /path/to/MoE-Inference-Acceleration/results/granite31_relative_rms_k630_v1_20261109/calibration/relative-tables.pt \
  relative4 /data/relative4.safetensors
```

The converter uses `torch.load(weights_only=True)` and writes a versioned
safetensors artifact. It folds the calibrated support rules into eligibility and
takes the square root of the saved second moment once. It preserves the saved
gate boundaries. It accepts the interval/pooled RMS bundles, including
`relative4`, `relative0` and `relative_mix`. The mixed variant's moments have
already been combined during calibration.

The artifact contains three tensors:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `scale` | `[layers, experts, top_k, bins]` | Square root of the calibrated moment, FP32 |
| `eligible` | Same as `scale` | Whether this assignment may be dropped, boolean |
| `edges` | `[layers, top_k, bins - 1]` | Increasing gate boundaries, FP32 |

Metadata includes `format=rms-moe-v1`, `weight_floor`, the selected variant and
the source bundle's SHA-256. A pooled table uses one bin and an empty final
dimension for `edges`. Shapes, types, finite scales and ordered edges are
validated at load time. Use calibration from the same checkpoint and router
precision as the deployed model. Matching dimensions alone does not establish
that a table is appropriate for a checkpoint.

For the local research variant A, a `[layer, expert, rank]` scale becomes
`scale[..., None]`, its support mask becomes `eligible[..., None]`, and
`edges` has shape `[layers, top_k, 0]`. Set metadata `weight_floor` to the JSON
value `null` when the calibration policy has no gate-floor protection, as in A.

## Enable pruning

The table path must be accessible to every worker. `budget` is the calibrated
fraction of score mass permitted to be removed at each token and layer.

```bash
vllm serve ibm-granite/granite-3.1-3b-a800m-instruct \
  --dtype bfloat16 \
  --hf-overrides '{"rms_expert_pruning":{"tables":"/data/relative4.safetensors","budget":0.09856785088777542}}'
```

A Gemma 4 MoE checkpoint takes the same option with its own calibration:

```bash
vllm serve google/gemma-4-26B-A4B-it --language-model-only \
  --dtype bfloat16 --moe-backend triton \
  --hf-overrides '{"rms_expert_pruning":{"tables":"/data/gemma4-rms.safetensors","budget":0.1}}'
```

For this checkpoint the scale shape is `[30, 128, 8, bins]`. In the generic
path, rank is the slot emitted by the native router. Collect calibration from
that same slot order and use the magnitude of the **final** routing weight for
binning and scoring. Gemma folds its learned per-expert scale into this
weight. The retained slots keep their original signed weights, ordering and
precision. Pre-existing padding IDs are excluded from table lookups. A zero
budget passes the native routing tensors through. Fused shared/zero expert
slots and input-weighted experts require their own integration. A shared MLP
computed separately from routed experts is outside the pruning budget.

The corresponding Python configuration is:

```python
from vllm import LLM

llm = LLM(
    model="ibm-granite/granite-3.1-3b-a800m-instruct",
    dtype="bfloat16",
    hf_overrides={
        "rms_expert_pruning": {
            "tables": "/data/relative4.safetensors",
            "budget": 0.09856785088777542,
        }
    },
)
```

The frozen budget for `relative_mix` is `0.09939710050821304`; for `relative0`
it is `0.10020879656076431`. These belong to the research calibrations at mean
K around 6.3. Actual K depends on the workload. The runtime performs no threshold
fitting.

## Routing and dispatch

The Granite router selects the original top-k logits, computes their FP32 softmax, and
rounds the gates to the hidden-state dtype before looking up the table. This
matches the research Granite router's BF16 gate convention. The interval index
counts boundaries strictly below the rounded gate. At rank r for expert e,

$$s_i = g_i\,\mathrm{scale}[\ell,e,r,b(g_i)].$$

Eligible scores are sorted in ascending order, with original router rank breaking
ties. Protected assignments follow them. The router drops the longest prefix
whose cumulative score is at most `budget * sum(scores)`, retaining at least
one expert. Finite protected scores participate in the denominator. A zero
budget or zero score sum retains every original assignment. The remaining
gates keep their original values.

On Granite CUDA, contiguous rows with at most 64 experts and top-k at most 32 fuse
selection, native rank restoration, softmax and pruning in one Triton launch.
The selection preserves CUDA radix top-k's input order, including signed zero,
and the final small-sort tie behavior required by rank-indexed calibration.
Larger or non-contiguous rows retain native top-k followed by fused pruning. The softmax uses libdevice
exponential and rounded
division to preserve the native gates. Stable ordering packs each non-negative
score's FP32 bits and its original rank into an integer key. Larger top-k uses
the native softmax followed by the fused pruning kernel. Dropped routes receive
expert ID `-1` and
weight zero. The original `[tokens, top_k]` shapes remain suitable for CUDA
graphs. Triton expert assignment omits invalid routes from aligned GEMMs.
Both projections use the existing grouped Triton kernels. The final reduction masks
invalid slots so unwritten workspace entries cannot contribute, even when they
contain stale NaNs. All expert weights remain resident.

In the general Triton path, RMS enables `skip_invalid_rows` on both expert
GEMMs. An invalid assignment returns before input/weight loads and does not
zero-fill its workspace row. SiLU also checks the assignment before reading
gate/up values or writing an output. This reuses the existing pad-aware
activation kernel with the native intermediate SiLU rounding. In this RMS-only
rounding mode the column tile fits the expert width (up to 1024 elements), and
each assignment receives its own CTA rather than reusing a grid capped at 256
rows. Both expert projections continue to use the existing grouped GEMMs;
no separate expert-computation kernels are introduced. These options
are confined to `mask_pruned_routes`; the default expert path retains its
original zero-fill and activation behavior. Grouped GEMMs can still execute
padded tile rows, and a retained assignment to the same expert still needs
that expert's weights. Fewer assignments do not guarantee proportional
memory traffic or end-to-end speedup.

The Granite integration uses the existing `custom_routing_function` extension.
It enables `FusedMoEConfig.mask_pruned_routes` explicitly for RMS. This flag
defaults to false: other callers retain their original backend candidates and
the original Triton reduction, including its expert-parallel branch. The oracle
requires `supports_pruned_routes()` only when this flag is true.
Without `rms_expert_pruning`, Granite does not load
the RMS module or tables and uses its original router and expert execution.

On Granite, budget zero still enables the RMS router and its calibrated gate rounding. To
disable the feature completely, omit `rms_expert_pruning`; budget zero is not a
promise of bitwise equality with the standard vLLM router.

## Validate on NVIDIA GPU

Run correctness checks before measuring performance:

```bash
pytest -q tests/model_executor/layers/test_rms_moe.py
pytest -q tests/kernels/moe/test_rms_pruning.py
pytest -q tests/kernels/moe/test_moe.py \
  -k 'default_reduction_preserves_upstream_path or triton_sum_ignores_pruned or triton_experts_execute_pruned'
pytest -q tests/models/language/generation/test_granite.py \
  -k rms_artifact
pytest -q tests/kernels/moe/test_unquantized_backend_selection.py -k pruning
pytest -q tests/models/language/generation/test_granite.py -k rms_backend
```

The tests cover interval boundaries, ties, protected assignments, zero budget,
native gate rounding, CUDA kernel parity, CUDA graph replay, unwritten slots,
tiny decode, aligned prefill and loading a table through `LLM` with real weights.
The LLM test compares a production MoE layer with an explicit PyTorch expert sum
with RMS disabled, with zero budget, and with pruning enabled. Reduction tests
also compare the disabled path exactly with the original implementation, with
and without an expert map and route IDs. The RMS GPU router suite runs on CUDA;
the artifact and policy tests also run on CPU.
The LLM backend test compares RMS generation under `auto` with the explicit
Triton RMS path.
The production precision and kernels may differ numerically from the historical
Transformers evaluation; repeat the quality benchmarks with the deployed vLLM.

Measure three configurations using the same Triton backend, model revision,
dtype, input lengths, concurrency, KV-cache limits and graph settings:

1. Standard vLLM routing, without `rms_expert_pruning`.
2. RMS routing with budget zero, to expose the alternate router's overhead.
3. RMS routing with the frozen nonzero budget.

For example, run a baseline latency measurement:

```bash
vllm bench latency \
  --model ibm-granite/granite-3.1-3b-a800m-instruct \
  --dtype bfloat16 --moe-backend triton --max-model-len 4096 \
  --input-len 512 --output-len 128 --batch-size 8 \
  --num-iters-warmup 5 --num-iters 30 --output-json baseline.json
```

Repeat with `--hf-overrides` as above and a distinct output file, then repeat
with budget zero. Also measure batch sizes 1, 32 and 128, prefill with long input
and one generated token, and decode with short input and many generated tokens.
Run repeated trials with alternating order and report latency distributions.
Use `vllm bench serve` for TTFT, inter-token latency and throughput under load.
Profile a separate run to confirm fewer expert GEMM rows/blocks and account for
the top-k, pruning and alignment costs. Padding to GEMM block sizes can absorb
some of the theoretical savings, especially with small batches.

MPS cannot execute this CUDA/Triton path. CPU checks establish table and policy
semantics; CUDA graphs, expert kernels, full model accuracy and speedup require
the GPU runs above. A lower average K alone does not establish a latency gain.
