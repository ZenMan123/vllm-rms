# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Calibrated RMS routing with a per-token dropped-score budget."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

RMS_MOE_FORMAT = "rms-moe-v1"


def validate_rms_budget(budget: float) -> float:
    if (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not math.isfinite(budget)
        or not 0 <= budget <= 1
    ):
        raise ValueError("RMS budget must be finite and between 0 and 1")
    return float(budget)


@dataclass
class RMSMoETables:
    """Frozen scales and eligibility, indexed by layer, expert, rank and bin.

    `scale` already contains the square root of the calibrated moment. A
    one-bin table represents pooled RMS. Eligibility includes every calibration
    support/fallback rule; protected cells still contribute to the score sum.
    """

    scale: torch.Tensor
    eligible: torch.Tensor
    edges: torch.Tensor
    weight_floor: float | None = 1e-8

    def __post_init__(self) -> None:
        if self.scale.ndim != 4 or any(d <= 0 for d in self.scale.shape):
            raise ValueError("RMS scale must have shape [layers, experts, top_k, bins]")
        layers, experts, top_k, bins = self.scale.shape
        if top_k > experts:
            raise ValueError("RMS top_k must not exceed the number of experts")
        if self.scale.dtype != torch.float32 or self.edges.dtype != torch.float32:
            raise ValueError("RMS scales and edges must be float32")
        if self.eligible.dtype != torch.bool or self.eligible.shape != self.scale.shape:
            raise ValueError("RMS eligibility must be boolean with the scale shape")
        if self.edges.shape != (layers, top_k, bins - 1):
            raise ValueError("RMS edges must have shape [layers, top_k, bins - 1]")
        if not torch.isfinite(self.scale).all() or (self.scale < 0).any():
            raise ValueError("RMS scales must be finite and non-negative")
        if not torch.isfinite(self.edges).all() or (self.edges < 0).any():
            raise ValueError("RMS edges must be finite and non-negative")
        if (self.edges[..., 1:] < self.edges[..., :-1]).any():
            raise ValueError("RMS edges must be sorted in ascending order")
        if self.weight_floor is not None and (
            isinstance(self.weight_floor, bool)
            or not isinstance(self.weight_floor, (int, float))
            or not math.isfinite(self.weight_floor)
            or self.weight_floor < 0
        ):
            raise ValueError("RMS weight_floor must be null or finite and non-negative")

    @classmethod
    def load(cls, path: str | Path) -> "RMSMoETables":
        """Load a local, versioned safetensors artifact on the CPU."""
        with safe_open(path, framework="pt", device="cpu") as file:
            metadata = file.metadata() or {}
            if metadata.get("format") != RMS_MOE_FORMAT:
                raise ValueError(f"Expected RMS artifact format {RMS_MOE_FORMAT}")
            if set(file.keys()) != {"scale", "eligible", "edges"}:
                raise ValueError("RMS artifact must contain scale, eligible and edges")
            try:
                weight_floor = json.loads(metadata["weight_floor"])
            except (KeyError, ValueError) as exc:
                raise ValueError("RMS weight_floor metadata must be JSON") from exc
            return cls(
                scale=file.get_tensor("scale"),
                eligible=file.get_tensor("eligible"),
                edges=file.get_tensor("edges"),
                weight_floor=weight_floor,
            )


def rms_keep_mask(
    ids: torch.Tensor,
    weights: torch.Tensor,
    scale: torch.Tensor,
    eligible: torch.Tensor,
    edges: torch.Tensor,
    budget: float,
    weight_floor: float | None,
) -> torch.Tensor:
    """Reference selection for routes sorted by their original router rank."""
    if budget == 0:
        return torch.ones_like(ids, dtype=torch.bool)
    rank = torch.arange(ids.shape[-1], device=ids.device)
    weights = weights.float()
    bins = (weights.unsqueeze(-1) > edges).sum(-1)
    scores = weights * scale[ids, rank, bins]
    reliable = eligible[ids, rank, bins] & torch.isfinite(weights)
    reliable &= torch.isfinite(scores)
    if weight_floor is not None:
        reliable &= weights > weight_floor
    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
    denominator = scores.sum(-1, keepdim=True)
    ranked = torch.where(reliable, scores, torch.inf)
    order = torch.argsort(ranked, dim=-1, stable=True)
    costs = ranked.gather(-1, order).cumsum(-1) / denominator.clamp_min(1e-30)
    drop = (costs <= budget) & (denominator > 0) & (rank < ids.shape[-1] - 1)
    return ~torch.zeros_like(drop).scatter(-1, order, drop)


class RMSMoERouter(nn.Module):
    """Native top-k softmax followed by calibrated RMS route pruning.

    Dropped slots have expert ID -1 and weight zero. Consumers must exclude
    them from both expert dispatch and the final reduction. The fixed top-k
    shape supports CUDA graph replay while the active assignments vary.
    """

    def __init__(
        self,
        tables: RMSMoETables,
        layer: int,
        budget: float,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        budget = validate_rms_budget(budget)
        if not 0 <= layer < tables.scale.shape[0]:
            raise ValueError("RMS layer index is outside the calibration table")
        self.budget = budget
        self.weight_floor = tables.weight_floor
        self.top_k = tables.scale.shape[2]
        self.num_experts = tables.scale.shape[1]
        for name in ("scale", "eligible", "edges"):
            self.register_buffer(
                name,
                getattr(tables, name)[layer].to(device).contiguous(),
                persistent=False,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if topk != self.top_k or gating_output.shape[-1] != self.num_experts:
            raise ValueError("Router dimensions do not match RMS calibration")
        if not renormalize:
            raise ValueError("RMS routing requires the original top-k softmax")
        if (
            gating_output.is_cuda
            and gating_output.is_contiguous()
            and self.num_experts <= 64
            and topk <= 32
        ):
            from vllm.model_executor.layers.fused_moe.router.rms_pruning import (
                fused_topk_rms_routes,
            )

            return fused_topk_rms_routes(
                gating_output,
                self.scale,
                self.eligible,
                self.edges,
                self.budget,
                self.weight_floor,
                hidden_states.dtype,
            )
        fuse_topk_sort = gating_output.is_cuda and topk <= 32
        logits, ids = torch.topk(gating_output, topk, dim=-1, sorted=not fuse_topk_sort)
        if fuse_topk_sort:
            from vllm.model_executor.layers.fused_moe.router.rms_pruning import (
                softmax_prune_rms_routes,
            )

            return softmax_prune_rms_routes(
                ids,
                logits,
                self.scale,
                self.eligible,
                self.edges,
                self.budget,
                self.weight_floor,
                hidden_states.dtype,
                sort_inputs=True,
            )
        weights = torch.softmax(logits, dim=-1, dtype=torch.float32)
        if ids.is_cuda and self.budget > 0:
            from vllm.model_executor.layers.fused_moe.router.rms_pruning import (
                prune_rms_routes,
            )

            return prune_rms_routes(
                ids,
                weights,
                self.scale,
                self.eligible,
                self.edges,
                self.budget,
                self.weight_floor,
                round_dtype=hidden_states.dtype,
            )
        weights = weights.to(hidden_states.dtype).float()
        if self.budget == 0:
            return weights, ids.int()
        keep = rms_keep_mask(
            ids,
            weights,
            self.scale,
            self.eligible,
            self.edges,
            self.budget,
            self.weight_floor,
        )
        return weights.masked_fill(~keep, 0), ids.masked_fill(~keep, -1).int()
