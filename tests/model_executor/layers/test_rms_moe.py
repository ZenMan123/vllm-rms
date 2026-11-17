# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for calibrated routing, independent of the expert backend."""

import pytest
import torch
from safetensors.torch import save_file

from examples.offline_inference.moe.export_rms_tables import export_tables
from vllm.model_executor.layers.rms_moe import (
    RMS_MOE_FORMAT,
    RMSMoERouter,
    RMSMoETables,
    rms_keep_mask,
)


def make_tables(top_k=8, bins=1):
    shape = (1, 8, top_k, bins)
    return RMSMoETables(
        torch.ones(shape),
        torch.ones(shape, dtype=torch.bool),
        torch.full((1, top_k, bins - 1), 0.125),
    )


@pytest.mark.parametrize("budget, removed", [(0.0, 0), (0.25, 2), (1.0, 7)])
def test_budget_keeps_one_route_and_breaks_score_ties_by_rank(budget, removed):
    tables = make_tables()
    ids = torch.arange(8).unsqueeze(0)
    weights = torch.full((1, 8), 0.125)
    keep = rms_keep_mask(
        ids,
        weights,
        tables.scale[0],
        tables.eligible[0],
        tables.edges[0],
        budget,
        tables.weight_floor,
    )
    assert keep.tolist() == [[i >= removed for i in range(8)]]


def test_protected_score_contributes_to_budget_but_cannot_be_removed():
    tables = make_tables()
    tables.eligible[0, 0] = False
    weights = torch.tensor(
        [[0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.0078125]]
    )
    ids = torch.arange(8).unsqueeze(0)
    keep = rms_keep_mask(
        ids,
        weights,
        tables.scale[0],
        tables.eligible[0],
        tables.edges[0],
        0.25,
        tables.weight_floor,
    )
    assert keep.tolist() == [[True, True, False, False, False, False, False, False]]


def test_bin_boundary_is_strict_and_unsupported_bin_is_protected():
    tables = make_tables(bins=2)
    tables.eligible[..., 0] = False
    weights = torch.full((2, 8), 0.125)
    weights[1, 0] = torch.nextafter(weights[1, 0], torch.tensor(1.0))
    ids = torch.arange(8).expand(2, -1)
    keep = rms_keep_mask(
        ids,
        weights,
        tables.scale[0],
        tables.eligible[0],
        tables.edges[0],
        1.0,
        tables.weight_floor,
    )
    assert keep[0].all()
    assert keep[1].tolist() == [False, True, True, True, True, True, True, True]


def test_zero_scores_and_nonfinite_routes_are_retained():
    tables = make_tables()
    tables.scale.zero_()
    ids = torch.arange(8).expand(2, -1)
    weights = torch.full((2, 8), 0.125)
    weights[1, 0] = float("nan")
    weights[1, 1] = float("inf")
    keep = rms_keep_mask(
        ids,
        weights,
        tables.scale[0],
        tables.eligible[0],
        tables.edges[0],
        1.0,
        tables.weight_floor,
    )
    assert keep.all()


def test_pooled_policy_can_drop_zero_gate_without_a_gate_floor():
    tables = make_tables()
    ids = torch.arange(8).unsqueeze(0)
    weights = torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0]])
    keep = rms_keep_mask(
        ids,
        weights,
        tables.scale[0],
        tables.eligible[0],
        tables.edges[0],
        0.1,
        None,
    )
    assert keep.tolist() == [[True, False, False, False, False, False, False, False]]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("budget", [0.0, 0.1, 1.0])
def test_router_preserves_native_rounded_gates(dtype, budget):
    torch.manual_seed(42)
    router = RMSMoERouter(make_tables(top_k=4), 0, budget, "cpu")
    logits = torch.randn(17, 8, dtype=dtype)
    hidden = torch.empty(17, 16, dtype=dtype)
    native_logits, native_ids = logits.topk(4, sorted=True)
    native = native_logits.softmax(-1, dtype=torch.float32).to(dtype).float()
    weights, ids = router(hidden, logits, 4, True)
    keep = ids >= 0
    assert torch.equal(ids[keep], native_ids[keep].int())
    assert torch.equal(weights[keep], native[keep])
    assert (weights[~keep] == 0).all()
    assert keep.any(-1).all()
    if budget == 0:
        assert keep.all()
    if budget == 1:
        assert (keep.sum(-1) == 1).all()
        assert (weights.sum(-1) < 1).all()


def test_empty_batch_and_single_expert_remain_well_defined():
    router = RMSMoERouter(make_tables(top_k=1), 0, 1.0, "cpu")
    w, ids = router(torch.empty(0, 16), torch.empty(0, 8), 1, True)
    assert w.shape == ids.shape == (0, 1)
    w, ids = router(torch.empty(2, 16), torch.randn(2, 8), 1, True)
    assert torch.equal(w, torch.ones(2, 1))
    assert (ids >= 0).all()


def test_artifact_roundtrip_and_shape_validation(tmp_path):
    tables = make_tables(bins=4)
    path = tmp_path / "rms.safetensors"
    tensors = {n: getattr(tables, n) for n in ("scale", "eligible", "edges")}
    metadata = {"format": RMS_MOE_FORMAT, "weight_floor": "1e-8"}
    save_file(tensors, path, metadata=metadata)
    loaded = RMSMoETables.load(path)
    assert torch.equal(loaded.scale, tables.scale)
    assert torch.equal(loaded.eligible, tables.eligible)
    tensors["edges"] = torch.zeros(1, 8, 2)
    save_file(tensors, path, metadata=metadata)
    with pytest.raises(ValueError, match="edges must have shape"):
        RMSMoETables.load(path)


@pytest.mark.parametrize(
    "budget", [-0.1, 1.1, float("nan"), float("inf"), True, "0.1", None]
)
def test_invalid_budget_rejected_before_inference(budget):
    with pytest.raises(ValueError, match="budget"):
        RMSMoERouter(make_tables(), 0, budget, "cpu")


@pytest.mark.parametrize("protect_bins", [False, True])
@pytest.mark.parametrize("weight_floor", [None, 1e-8])
def test_export_preserves_moments_and_calibration_support(
    tmp_path, protect_bins, weight_floor
):
    source = tmp_path / "research.pt"
    output = tmp_path / "runtime.safetensors"
    second = torch.tensor([4.0, 9.0]).expand(1, 16, 2).contiguous()
    reliable = torch.ones(1, 16, dtype=torch.bool)
    reliable[0, 0] = False
    bin_supported = torch.ones_like(second, dtype=torch.bool)
    bin_supported[..., 1] = False
    table = dict(
        statistic="rms",
        second=second,
        edges=torch.full((1, 2, 1), 0.25),
        reliable=reliable,
        bin_supported=bin_supported,
        protect_unsupported_bins=protect_bins,
        weight_floor=weight_floor,
    )
    torch.save(dict(experts=8, k=2, variants={"relative": table}), source)
    export_tables(source, "relative", output)
    loaded = RMSMoETables.load(output)
    assert loaded.weight_floor == weight_floor
    assert torch.equal(loaded.scale, second.sqrt().reshape(1, 8, 2, 2))
    expected = reliable[..., None].expand_as(second)
    if protect_bins:
        expected = expected & bin_supported
    assert torch.equal(loaded.eligible, expected.reshape(1, 8, 2, 2))


@pytest.mark.parametrize("metadata_value", [None, "not-json", '"text"', "true"])
def test_invalid_floor_metadata_has_clear_artifact_error(tmp_path, metadata_value):
    tables = make_tables()
    path = tmp_path / "invalid.safetensors"
    metadata = {"format": RMS_MOE_FORMAT}
    if metadata_value is not None:
        metadata["weight_floor"] = metadata_value
    save_file(
        {name: getattr(tables, name) for name in ("scale", "eligible", "edges")},
        path,
        metadata=metadata,
    )
    with pytest.raises(ValueError, match="weight_floor"):
        RMSMoETables.load(path)
