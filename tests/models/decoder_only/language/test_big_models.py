"""Compare the outputs of HF and vLLM when using greedy sampling.

This tests bigger models and use half precision.

Run `pytest tests/models/test_big_models.py`.
"""
import pytest

from vllm.platforms import current_platform

from ...utils import check_logprobs_close, check_outputs_equal

MODELS = [
    "meta-llama/Llama-2-7b-hf",
    # "mistralai/Mistral-7B-v0.1",  # Tested by test_mistral.py
    # "Deci/DeciLM-7b",  # Broken
    # "tiiuae/falcon-7b",  # Broken
    "EleutherAI/gpt-j-6b",
    # "mosaicml/mpt-7b",  # Broken
    # "Qwen/Qwen1.5-0.5B"  # Broken,
]

if not current_platform.is_cpu():
    MODELS += [
        # fused_moe which not supported on CPU
        "openbmb/MiniCPM3-4B",
        # Head size isn't supported on CPU
        "h2oai/h2o-danube3-4b-base",
    ]

# TODO: remove this after CPU float16 support ready
target_dtype = "float" if current_platform.is_cpu() else "half"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_tokens", [32])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:

    if model == "openbmb/MiniCPM3-4B":
        # the output becomes slightly different when upgrading to
        # pytorch 2.5 . Changing to logprobs checks instead of exact
        # output checks.
        NUM_LOG_PROBS = 8
        with hf_runner(model, dtype=dtype) as hf_model:
            hf_outputs = hf_model.generate_greedy_logprobs_limit(
                example_prompts, max_tokens, NUM_LOG_PROBS)

        with vllm_runner(model, dtype=dtype, enforce_eager=True) as vllm_model:
            vllm_outputs = vllm_model.generate_greedy_logprobs(
                example_prompts, max_tokens, NUM_LOG_PROBS)

        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )
    else:
        with hf_runner(model, dtype=dtype) as hf_model:
            hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

        with vllm_runner(model, dtype=dtype, enforce_eager=True) as vllm_model:
            vllm_outputs = vllm_model.generate_greedy(example_prompts,
                                                      max_tokens)

        check_outputs_equal(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", [target_dtype])
def test_model_print(
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    with vllm_runner(model, dtype=dtype, enforce_eager=True) as vllm_model:
        # This test is for verifying whether the model's extra_repr
        # can be printed correctly.
        print(vllm_model.model.llm_engine.model_executor.driver_worker.
              model_runner.model)
