# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU parity and graph replay for the RMS table selector."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.router.rms_pruning import (
    fused_topk_rms_routes,
    prune_rms_routes,
    softmax_prune_rms_routes,
)
from vllm.model_executor.layers.rms_moe import rms_keep_mask
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="RMS expert pruning requires CUDA"
)


@pytest.mark.parametrize("tokens", [0, 1, 17, 257])
@pytest.mark.parametrize("bins", [1, 4])
@pytest.mark.parametrize("top_k", [1, 6, 8])
@pytest.mark.parametrize("budget", [0.0, 0.1, 1.0])
@pytest.mark.parametrize("weight_floor", [None, 1e-8])
@pytest.mark.parametrize("round_dtype", [None, torch.bfloat16, torch.float16])
def test_rms_cuda_matches_reference(
    tokens, bins, top_k, budget, weight_floor, round_dtype
):
    torch.manual_seed(17)
    logits = torch.randn(tokens, 40, device="cuda", dtype=torch.bfloat16)
    values, ids = logits.topk(top_k)
    weights = values.softmax(-1, dtype=torch.float32)
    if round_dtype is None:
        weights = weights.bfloat16().float()
    rounded_weights = (
        weights if round_dtype is None else weights.to(round_dtype).float()
    )
    scale = torch.rand(40, top_k, bins, device="cuda") + 0.1
    eligible = torch.rand(40, top_k, bins, device="cuda") > 0.2
    edges = torch.linspace(0, 1, bins + 1, device="cuda")[1:-1]
    edges = edges.expand(top_k, -1).contiguous()
    keep = rms_keep_mask(
        ids, rounded_weights, scale, eligible, edges, budget, weight_floor
    )
    out_weights, out_ids = prune_rms_routes(
        ids, weights, scale, eligible, edges, budget, weight_floor, round_dtype
    )
    assert torch.equal(out_ids, ids.masked_fill(~keep, -1).int())
    assert torch.equal(out_weights, rounded_weights.masked_fill(~keep, 0))


@pytest.mark.parametrize("tokens", [0, 1, 17, 257])
@pytest.mark.parametrize("top_k", [1, 3, 6, 8, 16, 32])
@pytest.mark.parametrize("budget", [0.0, 0.1, 1.0])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_fused_softmax_rms_preserves_native_gates(tokens, top_k, budget, dtype):
    torch.manual_seed(129)
    logits = torch.randn(tokens, 40, device="cuda", dtype=dtype)
    values, ids = logits.topk(top_k)
    weights = values.softmax(-1, dtype=torch.float32).to(dtype).float()
    scale = torch.rand(40, top_k, 4, device="cuda") + 0.1
    eligible = torch.rand(40, top_k, 4, device="cuda") > 0.2
    edges = torch.tensor([0.04, 0.08, 0.16], device="cuda")
    edges = edges.expand(top_k, -1).contiguous()
    keep = rms_keep_mask(ids, weights, scale, eligible, edges, budget, 1e-8)
    actual_weights, actual_ids = softmax_prune_rms_routes(
        ids, values, scale, eligible, edges, budget, 1e-8, dtype
    )
    assert torch.equal(actual_ids, ids.masked_fill(~keep, -1).int())
    assert torch.equal(actual_weights, weights.masked_fill(~keep, 0))


@pytest.mark.parametrize("tokens", [1, 17, 513, 0])
@pytest.mark.parametrize(
    "num_experts, top_k",
    [
        (16, 1),
        (16, 3),
        (16, 8),
        (40, 1),
        (40, 3),
        (40, 8),
        (40, 31),
        (64, 1),
        (64, 8),
        (64, 31),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("ties", [False, True])
def test_unsorted_rms_topk_preserves_sorted_route_ranks(
    tokens, num_experts, top_k, dtype, ties
):
    """Fusing the final top-k sort must preserve rank-indexed calibration."""
    torch.manual_seed(731)
    logits = torch.randn(tokens, num_experts, device="cuda", dtype=dtype)
    if ties:
        logits = logits.round()
    scale = torch.rand(num_experts, top_k, 4, device="cuda") + 0.1
    eligible = torch.rand_like(scale) > 0.2
    edges = torch.tensor([0.04, 0.08, 0.16], device="cuda")
    edges = edges.expand(top_k, -1).contiguous()
    values, ids = logits.topk(top_k, sorted=True)
    expected = softmax_prune_rms_routes(
        ids, values, scale, eligible, edges, 0.1, None, dtype
    )
    values, ids = logits.topk(top_k, sorted=False)
    actual = softmax_prune_rms_routes(
        ids, values, scale, eligible, edges, 0.1, None, dtype, sort_inputs=True
    )
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, atol=0, rtol=0)

    fused = fused_topk_rms_routes(logits, scale, eligible, edges, 0.1, None, dtype)
    for result, reference in zip(fused, expected):
        torch.testing.assert_close(result, reference, atol=0, rtol=0)


def test_fused_softmax_rms_graph_replay():
    ids = torch.arange(8, device="cuda").expand(4, -1).contiguous()
    logits = torch.zeros((4, 8), device="cuda", dtype=torch.bfloat16)
    scale = torch.ones(8, 8, 2, device="cuda")
    eligible = torch.ones_like(scale, dtype=torch.bool)
    eligible[0, 0, 0] = False
    edges = torch.full((8, 1), 0.125, device="cuda")
    args = (ids, logits, scale, eligible, edges, 0.25, 1e-8, logits.dtype)
    softmax_prune_rms_routes(*args)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_weights, actual_ids = softmax_prune_rms_routes(*args)
    for row in range(-1, 4):
        if row >= 0:
            logits[row, 0] = 0.5
        graph.replay()
        weights = logits.softmax(-1, dtype=torch.float32).bfloat16().float()
        keep = rms_keep_mask(ids, weights, scale, eligible, edges, 0.25, 1e-8)
        assert torch.equal(actual_ids, ids.masked_fill(~keep, -1).int())
        assert torch.equal(actual_weights, weights.masked_fill(~keep, 0))


def test_rms_replay_changes_masks_and_respects_exact_boundaries():
    ids = torch.arange(8, device="cuda").expand(4, -1).contiguous()
    weights = torch.full((4, 8), 0.125, device="cuda")
    scale = torch.ones(8, 8, 2, device="cuda")
    eligible = torch.ones_like(scale, dtype=torch.bool)
    edges = torch.full((8, 1), 0.125, device="cuda")
    eligible[0, 0, 0] = False
    args = (ids, weights, scale, eligible, edges, 0.25, 1e-8)
    prune_rms_routes(*args)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_weights, actual_ids = prune_rms_routes(*args)
    for row in range(4):
        weights[row, 0] = torch.nextafter(weights[row, 0], weights.new_tensor(1.0))
        graph.replay()
        expected = rms_keep_mask(*args)
        assert torch.equal(actual_ids, ids.masked_fill(~expected, -1).int())
        assert torch.equal(actual_weights, weights.masked_fill(~expected, 0))
