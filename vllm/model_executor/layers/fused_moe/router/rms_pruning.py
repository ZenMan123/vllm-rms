# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse native gate math, stable score ordering and the RMS budget into one launch."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _sort_small_topk(ids, logits, rank, valid):
    # CUDA top-k's small final sort uses a 32-item bitonic network. Preserve
    # its exchanges of equal keys: calibration is indexed by the resulting rank.
    for level in tl.static_range(1, 6):
        for step in tl.static_range(level - 1, -1, -1):
            distance = 1 << step
            partner = rank ^ distance
            other_logits = tl.gather(logits, partner, 0)
            other_ids = tl.gather(ids, partner, 0)
            other_valid = tl.gather(valid, partner, 0)
            low = (rank & distance) == 0
            a = tl.where(low, logits, other_logits)
            b = tl.where(low, other_logits, logits)
            valid_a = tl.where(low, valid, other_valid)
            valid_b = tl.where(low, other_valid, valid)
            greater = ((a != a) & (b == b)) | (a > b)
            direction = (rank & (1 << level)) != 0
            swap = ((greater & valid_a) | ~valid_b) == direction
            logits = tl.where(swap, other_logits, logits)
            ids = tl.where(swap, other_ids, ids)
            valid = tl.where(swap, other_valid, valid)
    return ids, logits


@triton.jit
def _prune_rms_routes(
    ids_ptr,
    weights_ptr,
    scale_ptr,
    eligible_ptr,
    edges_ptr,
    out_ids_ptr,
    out_weights_ptr,
    TOP_K: tl.constexpr,
    BINS: tl.constexpr,
    BUDGET: tl.constexpr,
    WEIGHT_FLOOR: tl.constexpr,
    ROUND_DTYPE: tl.constexpr,
    APPLY_SOFTMAX: tl.constexpr,
    BLOCK: tl.constexpr,
    SORT_INPUTS: tl.constexpr,
    NUM_EXPERTS: tl.constexpr = 0,
):
    token = tl.program_id(0)
    rank = tl.arange(0, BLOCK)
    valid = rank < TOP_K
    offsets = token * TOP_K + rank
    if NUM_EXPERTS:
        # Reproduce CUDA radix top-k's gather order: values above the kth
        # key first, then equal keys in input order. Preserve signed zero
        # in radix keys; the subsequent native bitonic sort uses float ties.
        expert_rank = tl.arange(0, 64)
        logits = tl.load(
            weights_ptr + token * NUM_EXPERTS + expert_rank,
            expert_rank < NUM_EXPERTS,
            other=-float("inf"),
        ).to(tl.float32)
        bits = logits.to(tl.uint32, bitcast=True)
        keys = tl.where(bits & 0x80000000 != 0, ~bits, bits ^ 0x80000000)
        keys = tl.where(logits != logits, 0xFFFFFFFF, keys)
        keys = tl.where(expert_rank < NUM_EXPERTS, keys, 0)
        sorted_keys = tl.sort(keys, descending=True)
        threshold = tl.sum(tl.where(expert_rank == TOP_K - 1, sorted_keys, 0), 0)
        above = (keys > threshold) & (expert_rank < NUM_EXPERTS)
        equal = (keys == threshold) & (expert_rank < NUM_EXPERTS)
        num_above = tl.sum(above.to(tl.int32), 0)
        above_pos = tl.cumsum(above.to(tl.int32), 0) - 1
        equal_pos = num_above + tl.cumsum(equal.to(tl.int32), 0) - 1
        pos = tl.where(above, above_pos, equal_pos)
        selected = above | (equal & (pos < TOP_K))
        packed = tl.where(selected, pos * 64 + expert_rank, 0x7FFFFFFF)
        packed = tl.sort(packed, descending=False)
        ids = tl.gather(packed, rank, 0) % 64
        ids = tl.where(valid, ids, 0)
        weights = tl.gather(logits, ids, 0)
        weights = tl.where(valid, weights, 0.0)
    else:
        ids = tl.load(ids_ptr + offsets, valid, other=0)
        weights = tl.load(weights_ptr + offsets, valid, other=0)
    if SORT_INPUTS:
        ids, weights = _sort_small_topk(ids, weights, rank, valid)
    if APPLY_SOFTMAX:
        logits = tl.where(valid, weights.to(tl.float32), -float("inf"))
        # Match the FP32 warp softmax's exp and rounded division. Approximate
        # exp2/reciprocal can move a native BF16 gate across a bin boundary.
        numerator = tl.extra.cuda.libdevice.exp(logits - tl.max(logits, 0))
        weights = tl.div_rn(numerator, tl.sum(numerator, 0))
    if ROUND_DTYPE is not None:
        # Preserve the calibrated native gate rounding without two standalone
        # dtype-conversion kernels and their intermediate buffers.
        weights = weights.to(ROUND_DTYPE).to(tl.float32)
    bins = tl.full((BLOCK,), 0, tl.int32)
    for boundary in tl.static_range(BINS - 1):
        edge = tl.load(edges_ptr + rank * (BINS - 1) + boundary, valid, other=0)
        bins += (weights > edge).to(tl.int32)
    cell = (ids * TOP_K + rank) * BINS + bins
    scale = tl.load(scale_ptr + cell, valid, other=0)
    eligible = tl.load(eligible_ptr + cell, valid, other=0)
    scores = weights * scale
    finite = (scores == scores) & (tl.abs(scores) < float("inf"))
    reliable = valid & eligible & finite
    if WEIGHT_FLOOR is not None:
        reliable &= weights > WEIGHT_FLOOR
    scores = tl.where(finite & valid, tl.maximum(scores, 0.0), 0.0)
    denominator = tl.sum(scores, 0)
    ranked = tl.where(reliable, scores, float("inf"))

    # Non-negative FP32 bit patterns sort numerically. Append the original
    # rank to make ties stable without materializing a K-by-K comparison.
    # Clear the sign bit so -0.0 and +0.0 have the same score key.
    bits = ranked.to(tl.uint32, bitcast=True) & 0x7FFFFFFF
    keys = bits.to(tl.uint64) * BLOCK + rank.to(tl.uint64)
    ordered_keys = tl.sort(keys, descending=False)
    original_rank = (ordered_keys % BLOCK).to(tl.int32)
    ordered = (ordered_keys // BLOCK).to(tl.uint32).to(tl.float32, bitcast=True)
    costs = tl.div_rn(tl.cumsum(ordered, 0), tl.maximum(denominator, 1e-30))
    drop = (
        (BUDGET > 0)
        & tl.gather(reliable, original_rank, 0)
        & (denominator > 0)
        & (rank < TOP_K - 1)
        & (costs <= BUDGET)
    )
    offsets = token * TOP_K + original_rank
    valid = original_rank < TOP_K
    ids = tl.gather(ids, original_rank, 0)
    weights = tl.gather(weights, original_rank, 0)
    tl.store(out_ids_ptr + offsets, tl.where(drop, -1, ids), valid)
    tl.store(out_weights_ptr + offsets, tl.where(drop, 0.0, weights), valid)


def _launch_rms_pruning(
    ids: torch.Tensor,
    weights: torch.Tensor,
    scale: torch.Tensor,
    eligible: torch.Tensor,
    edges: torch.Tensor,
    budget: float,
    weight_floor: float | None,
    round_dtype: torch.dtype | None,
    apply_softmax: bool,
    sort_inputs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if budget == 0 and not apply_softmax:
        if round_dtype is not None:
            weights = weights.to(round_dtype).float()
        return weights, ids.int()
    num_tokens, top_k = ids.shape
    out_ids = torch.empty_like(ids, dtype=torch.int32)
    out_weights = torch.empty_like(weights, dtype=torch.float32)
    if num_tokens:
        _prune_rms_routes[(num_tokens,)](
            ids,
            weights,
            scale,
            eligible,
            edges,
            out_ids,
            out_weights,
            TOP_K=top_k,
            BINS=scale.shape[-1],
            BUDGET=budget,
            WEIGHT_FLOOR=weight_floor,
            ROUND_DTYPE={
                torch.float16: tl.float16,
                torch.bfloat16: tl.bfloat16,
                torch.float32: tl.float32,
            }.get(round_dtype),
            APPLY_SOFTMAX=apply_softmax,
            BLOCK=32 if sort_inputs else triton.next_power_of_2(top_k),
            SORT_INPUTS=sort_inputs,
            num_warps=1,
            enable_fp_fusion=False,
        )
    return out_weights, out_ids


def prune_rms_routes(
    ids: torch.Tensor,
    weights: torch.Tensor,
    scale: torch.Tensor,
    eligible: torch.Tensor,
    edges: torch.Tensor,
    budget: float,
    weight_floor: float | None,
    round_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prune precomputed gates, optionally rounding to the native gate dtype."""
    return _launch_rms_pruning(
        ids, weights, scale, eligible, edges, budget, weight_floor, round_dtype, False
    )


def softmax_prune_rms_routes(
    ids: torch.Tensor,
    logits: torch.Tensor,
    scale: torch.Tensor,
    eligible: torch.Tensor,
    edges: torch.Tensor,
    budget: float,
    weight_floor: float | None,
    gate_dtype: torch.dtype,
    *,
    sort_inputs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse optional CUDA top-k sorting, softmax, gate rounding and pruning."""
    if ids.shape[-1] > 32:
        raise ValueError("Fused RMS softmax supports at most 32 selected experts")
    return _launch_rms_pruning(
        ids,
        logits,
        scale,
        eligible,
        edges,
        budget,
        weight_floor,
        gate_dtype,
        True,
        sort_inputs,
    )


def fused_topk_rms_routes(
    logits: torch.Tensor,
    scale: torch.Tensor,
    eligible: torch.Tensor,
    edges: torch.Tensor,
    budget: float,
    weight_floor: float | None,
    gate_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse native small-row top-k, gate rounding and calibrated pruning."""
    num_tokens, num_experts = logits.shape
    top_k = scale.shape[1]
    if (
        not logits.is_contiguous()
        or not 1 <= top_k <= min(32, num_experts)
        or num_experts > 64
    ):
        raise ValueError(
            "Fused RMS top-k requires contiguous rows, E <= 64 and K <= 32"
        )
    out_ids = torch.empty((num_tokens, top_k), device=logits.device, dtype=torch.int32)
    out_weights = torch.empty(
        (num_tokens, top_k), device=logits.device, dtype=torch.float32
    )
    if num_tokens:
        _prune_rms_routes[(num_tokens,)](
            None,
            logits,
            scale,
            eligible,
            edges,
            out_ids,
            out_weights,
            TOP_K=top_k,
            BINS=scale.shape[-1],
            BUDGET=budget,
            WEIGHT_FLOOR=weight_floor,
            ROUND_DTYPE={
                torch.bfloat16: tl.bfloat16,
                torch.float16: tl.float16,
                torch.float32: tl.float32,
            }[gate_dtype],
            APPLY_SOFTMAX=True,
            BLOCK=32,
            SORT_INPUTS=True,
            NUM_EXPERTS=num_experts,
            num_warps=1,
            enable_fp_fusion=False,
        )
    return out_weights, out_ids
