# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from functools import partial

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

# 지원하는 모든 모델의 디코더 레이어 타입을 포함
# Llama (1/2/3/TinyLlama), Mistral, Qwen2/3, Phi3/4
_SUPPORTED_DECODER_LAYERS = (
    LlamaDecoderLayer,
    MistralDecoderLayer,
    Qwen2DecoderLayer,
    Phi3DecoderLayer,
)

check_fn = lambda submodule: isinstance(submodule, _SUPPORTED_DECODER_LAYERS)


def apply_fsdp_checkpointing(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying fsdp activation checkpointing...")

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )
