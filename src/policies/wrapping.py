# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import functools

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from transformers.models.phi4.modeling_phi4 import Phi4DecoderLayer
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)


def get_size_policy(min_params=1e8):
    num_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=min_params
    )
    return num_wrap_policy


def get_llama_wrapper():
    """we register our main layer class and use the fsdp transformer wrapping policy
    ensures embedding layers are in the root fsdp unit for shared access and that fsdp units map to transformer layers
    """
    # ====   use new transformer wrapper

    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaDecoderLayer,
        },
    )

    return llama_auto_wrap_policy

def get_mistral_wrapper():
    mistral_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            MistralDecoderLayer,
        },
    )

    return mistral_auto_wrap_policy

def get_qwen_wrapper():
    qwen_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Qwen3DecoderLayer,
        },
    )

    return qwen_auto_wrap_policy

def get_phi_wrapper():
    phi_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Phi4DecoderLayer,
        },
    )

    return phi_auto_wrap_policy


def get_wrapper_by_model_name(model_name):
    """모델 이름으로 적절한 wrapping policy 자동 선택"""
    name_lower = model_name.lower()
    if "mistral" in name_lower:
        return get_mistral_wrapper()
    elif "qwen" in name_lower:
        return get_qwen_wrapper()
    elif "phi" in name_lower:
        return get_phi_wrapper()
    else:
        # Llama, TinyLlama 등 기본값
        return get_llama_wrapper()
