# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from safetensors.torch import save_file
from transformers import GraniteConfig

from vllm.model_executor.models.granite import granite_layer_attn_params
from vllm.platforms import current_platform

from ...utils import check_logprobs_close

MODELS = [
    # TODO(sang): Sliding window should be tested separately.
    "ibm/PowerLM-3b",
    "ibm/PowerMoE-3b",
    "ibm-granite/granite-swash-2b",
    "ibm-granite/granite-swash-3b-a600m",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.cpu_model
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs
        )

    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs
        )
    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


def test_granite_swa_features_are_off_without_swa_config():
    """A plain Granite config must not pick up sliding windows or sinks."""
    config = GraniteConfig(num_hidden_layers=4)
    theta = config.rope_parameters["rope_theta"]

    assert [granite_layer_attn_params(config, i) for i in range(4)] == [
        (None, theta, False)
    ] * 4


@pytest.mark.parametrize(
    "attention_sinks, expected_sink", [(None, True), (False, False)]
)
def test_granite_swa_features_resolve_per_layer(attention_sinks, expected_sink):
    """`layer_types`/`layer_rope_theta` apply per layer; theta 0 means NoPE."""
    kwargs = {} if attention_sinks is None else {"attention_sinks": attention_sinks}
    config = GraniteConfig(
        num_hidden_layers=3,
        sliding_window=128,
        layer_types=["full_attention", "sliding_attention", "sliding_attention"],
        layer_rope_theta=[10000.0, 0.0, 1000000.0],
        **kwargs,
    )

    assert [granite_layer_attn_params(config, i) for i in range(3)] == [
        (None, 10000.0, expected_sink),
        (128, 0.0, expected_sink),
        (128, 1000000.0, expected_sink),
    ]


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="RMS Granite integration requires CUDA"
)
@pytest.mark.parametrize("budget", [None, 0.0, 0.5])
def test_granite_rms_artifact_through_llm(vllm_runner, tmp_path, monkeypatch, budget):
    """Load tables through LLM and compare its MoE with explicit kept experts."""
    from vllm.model_executor.layers.rms_moe import RMS_MOE_FORMAT

    # The test sends its own inspection callback to the local engine worker.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    path = tmp_path / "rms.safetensors"
    save_file(
        {
            "scale": torch.ones(32, 40, 8, 1),
            "eligible": torch.ones(32, 40, 8, 1, dtype=torch.bool),
            "edges": torch.empty(32, 8, 0),
        },
        path,
        metadata={"format": RMS_MOE_FORMAT, "weight_floor": "1e-8"},
    )

    def check_loaded_moe(worker):
        from vllm.config import set_current_vllm_config
        from vllm.forward_context import set_forward_context
        from vllm.model_executor.layers.rms_moe import RMSMoERouter

        model = worker.get_model()
        moe = model.model.layers[0].block_sparse_moe
        routers = [m for m in moe.modules() if isinstance(m, RMSMoERouter)]
        routed = moe.experts.routed_experts
        assert routed.moe_config.mask_pruned_routes == (budget is not None)
        if budget is None:
            from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
                FusedTopKRouter,
            )

            assert not routers
            assert type(moe.experts.router) is FusedTopKRouter
        else:
            assert len(routers) == 1
            router = routers[0]
            assert router.budget == budget
            assert router.scale.is_cuda
        # Real checkpoint weights, real router, and the production expert path.
        from tests.kernels.utils import torch_experts

        torch.manual_seed(7)
        for num_tokens in (1, 8, 17):
            x = torch.randn(
                num_tokens, moe.hidden_size, device="cuda", dtype=torch.bfloat16
            )
            logits, _ = moe.gate(x)
            weights, ids = moe.experts.router.select_experts(x, logits)
            keep = ids >= 0
            assert keep.any(-1).all()
            assert keep.all() if budget in (None, 0) else (~keep).any()
            with (
                torch.inference_mode(),
                set_current_vllm_config(worker.vllm_config),
                set_forward_context(None, worker.vllm_config, num_tokens=x.shape[0]),
            ):
                reference = torch_experts(
                    x, routed.w13_weight, routed.w2_weight, weights, ids.clamp_min(0)
                )
                actual = moe(x)
            torch.testing.assert_close(actual, reference, atol=2e-2, rtol=2e-2)
        return True

    with vllm_runner(
        "ibm-granite/granite-3.1-3b-a800m-instruct",
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=8,
        max_num_batched_tokens=256,
        gpu_memory_utilization=0.10,
        kv_cache_memory_bytes=128 * 1024**2,
        enforce_eager=True,
        kernel_config={"moe_backend": "triton"},
        hf_overrides=(
            {}
            if budget is None
            else {"rms_expert_pruning": {"tables": str(path), "budget": budget}}
        ),
    ) as runner:
        assert all(runner.llm.collective_rpc(check_loaded_moe))
        outputs = runner.generate_greedy(["The capital of France is"], max_tokens=4)
        assert len(outputs) == 1 and len(outputs[0][0]) > 0


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="RMS Granite integration requires CUDA"
)
@pytest.mark.parametrize("backend", ["auto"])
def test_granite_rms_backend_matches_triton(vllm_runner, tmp_path, backend):
    """Backend selection must preserve the calibrated router and its output."""
    path = tmp_path / "rms.safetensors"
    save_file(
        {
            "scale": torch.ones(32, 40, 8, 1),
            "eligible": torch.ones(32, 40, 8, 1, dtype=torch.bool),
            "edges": torch.empty(32, 8, 0),
        },
        path,
        metadata={"format": "rms-moe-v1", "weight_floor": "null"},
    )
    results = []
    for candidate in ("triton", backend):
        with vllm_runner(
            "ibm-granite/granite-3.1-3b-a800m-instruct",
            dtype="bfloat16",
            max_model_len=128,
            max_num_seqs=8,
            max_num_batched_tokens=256,
            gpu_memory_utilization=0.10,
            kv_cache_memory_bytes=128 * 1024**2,
            enforce_eager=True,
            kernel_config={"moe_backend": candidate},
            hf_overrides={"rms_expert_pruning": {"tables": str(path), "budget": 0.25}},
        ) as runner:
            results.append(
                runner.generate_greedy_logprobs(
                    ["The capital of France is", "What is 17 plus 25?"],
                    max_tokens=8,
                    num_logprobs=5,
                )
            )
    check_logprobs_close(
        outputs_0_lst=results[0],
        outputs_1_lst=results[1],
        name_0="RMS Triton",
        name_1=f"RMS {backend}",
    )
