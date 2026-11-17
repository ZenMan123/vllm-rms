# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export the Granite RMS research tables to a runtime safetensors artifact."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from vllm.model_executor.layers.rms_moe import RMS_MOE_FORMAT, RMSMoETables


def export_tables(source: Path, variant: str, output: Path) -> None:
    bundle = torch.load(source, map_location="cpu", weights_only=True)
    table = bundle["variants"][variant]
    if table["statistic"] != "rms":
        raise ValueError("Only RMS moments can be exported")
    if (
        table.get("selection", "score") != "score"
        or table.get("denominator", "all") != "all"
    ):
        raise ValueError(
            "Only the standard score order and all-score budget are supported"
        )
    if table.get("edge_scope", "layer_rank") != "layer_rank":
        raise ValueError("Edges must be shared by experts at each layer and rank")
    experts, top_k = int(bundle["experts"]), int(bundle["k"])
    second = table["second"]
    layers, cells, bins = second.shape
    if cells != experts * top_k:
        raise ValueError("Moment table does not match the expert and rank dimensions")
    if not torch.isfinite(second).all() or (second < 0).any():
        raise ValueError("Calibrated second moments must be finite and non-negative")
    eligible = table["reliable"].unsqueeze(-1).expand_as(second)
    if table["protect_unsupported_bins"]:
        eligible = eligible & table["bin_supported"]
    shape = (layers, experts, top_k, bins)
    tables = RMSMoETables(
        scale=second.float().sqrt().reshape(shape).contiguous(),
        eligible=eligible.reshape(shape).contiguous(),
        edges=table["edges"].float().contiguous(),
        weight_floor=table["weight_floor"],
    )
    save_file(
        {name: getattr(tables, name) for name in ("scale", "eligible", "edges")},
        output,
        metadata={
            "format": RMS_MOE_FORMAT,
            "weight_floor": json.dumps(tables.weight_floor),
            "variant": variant,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("variant")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export_tables(args.source, args.variant, args.output)
