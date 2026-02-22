import os
import sys
from pkg_resources import packaging

import fire
import torch
import torch.distributed as dist
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_kbit_training
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    default_data_collator,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
 
from src.configs import fsdp_config, train_config, inference_config
from src.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from src.utils import fsdp_auto_wrap_policy
from src.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_subdirectories
)
from src.utils.dataset_utils import *
from src.inference.model_utils import load_model, load_peft_model
from src.utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies
)
from src.utils.tokenizer_utils import setup_tokenizer, resize_model_embeddings

from setproctitle import setproctitle


setproctitle("Tenemin")


#train_config 파일을 어디서 받아서 넘겨주는지 모르겠음.
def main(**kwargs):
    # Update the configuration for the training and sharding process
    update_config((train_config, fsdp_config, inference_config), **kwargs)
    train_cfg_ins = train_config()
    inference_cfg_ins = inference_config()
    update_config((train_cfg_ins, inference_cfg_ins), **kwargs)


    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    #FSDP사용 여부 확인 및 FSDP setup
    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print("local rank", local_rank)
        print("rank", rank)
        print("world size", world_size)

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    # Load the tokenizer and add special tokens
    # pretrain tokenizer load
    print("\n" + "="*60)
    print("🔧 Setting up tokenizer...")
    print("="*60)
    tokenizer, model_family = setup_tokenizer(train_config.model_name, padding_side='left')
    print("="*60 + "\n")
    dataset_config = generate_dataset_config(train_config, kwargs)

     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )
    # print("max input lenght:", dataset_train.getMaxLength())

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    #안걸림림
    if train_config.run_validation is True:
        dataset_val = get_preprocessed_dataset(
            tokenizer,
            dataset_config,
            split="test",
        )
        if not train_config.enable_fsdp or rank == 0:
            print(f"--> Validation Set Length = {len(dataset_val)}")

    train_sampler = None
    val_sampler = None
    if train_config.enable_fsdp:
        #DistributedSampler의 역할 --> 전체 데이터셋을 world_size(num_replicas)만큼으로 나누어 데이터가 중복 학습 방지지
        train_sampler = DistributedSampler(
            dataset_train,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=True,
        )
        #안걸림림
        if train_config.run_validation:
            val_sampler = DistributedSampler(
                dataset_val,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
            )

    # Create DataLoaders for the training and validation dataset
    # 학습용 데이터 로더 생성
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=False,
        collate_fn=llmweightst_dataset_collate_fn if 'llmweightst' in dataset_config.dataset else
                    llmscott_dataset_collate_fn if 'llmscott' in dataset_config.dataset else
                    krsl_dataset_collate_fn if 'krsl' in dataset_config.dataset else
                    llmcmt_dataset_collate_fn if 'llmcmt' in dataset_config.dataset else
                    llmmt_dataset_collate_fn if 'llmmt' in dataset_config.dataset else
                    llmstepst_dataset_collate_fn if 'llmstepst' in dataset_config.dataset else
                    llmst_dataset_collate_fn if 'llmst' in dataset_config.dataset else
                    dataset_collate_fn
    )
    
    #안걸림
    eval_dataloader = None
    if train_config.run_validation is True:
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=train_config.val_batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            sampler=val_sampler if val_sampler else None,
            drop_last=True,
            collate_fn=eval_dataset_collate_fn
        )
        

    # Load the pre-trained model and setup its configuration
    # use_cache = False or Non으로 강제,  use_cache=True --> False, use_cache=False --> None // None이 들어갈 경우 base model의 기본값이 들어감감
    use_cache = False if train_config.enable_fsdp else None
    # 안걸림 True, False
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        print("v.is_devrelease and v.dev >= 20230701", v.is_devrelease, v.dev)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        if rank == 0:
            model = AutoModelForCausalLM.from_pretrained(
            train_config.model_name,
            use_cache=use_cache,
            torch_dtype=torch.bfloat16,   # ✅ 1
            low_cpu_mem_usage=True,       # ✅ 2
        )

        else:
            llama_config = AutoConfig.from_pretrained(train_config.model_name)
            llama_config.use_cache = use_cache
            with torch.device("meta"):
                model = AutoModelForCausalLM(llama_config)
    # 여기 걸림
    else:
        model = AutoModelForCausalLM.from_pretrained(
            train_config.model_name,
            use_cache=use_cache,
        )
        
        # Embedding layer 크기 조정
        print("\n🔧 Checking token embeddings...")
        model = resize_model_embeddings(model, tokenizer, model_family)
        print("="*60 + "\n")
        # 안걸림
        if 'dpo' in train_config.dataset:
            ref_model = AutoModelForCausalLM.from_pretrained(
                train_config.ref_model_name,
                use_cache=use_cache,
            )
        # 여기 걸림
        else:
            ref_model = None
    # 안걸림 True, False
    if train_config.enable_fsdp and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model) 
            # model = model.to_bettertransformer()
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Prepare the model for int8 training if quantization is enabled
    # 안걸림
    if train_config.quantization:
        model = prepare_model_for__training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    # bf16
    # 안걸림 Ture, False
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)
    # for name, param in model.named_parameters():
    #     print(f"Parameter name: {name}, Type: {param.dtype}")
        # param.data = param.data.to(torch.float16)
    

    
    # 여기 걸림
    current_epoch = 0
    if train_config.use_peft:
        print("train cfg ckpt_continue:", train_config.ckpt_continue)
        # ckpt_continue가 존재하고, 'adapter_model.bin'이 있을 경우(adapter only)
        if train_config.ckpt_continue is not None and os.path.exists(os.path.join(train_config.ckpt_continue, 'adapter_model.bin')):
            model = load_peft_model(model, train_config.ckpt_continue)
            peft_config = generate_peft_config(train_config, kwargs)
            model = get_peft_model(model, peft_config)
            current_epoch = int(train_config.ckpt_continue.split('/')[-1].split('-')[-1])
        # ckpt_continue만 존재할 경우 (full model)
        elif train_config.ckpt_continue is not None and not os.path.exists(os.path.join(train_config.ckpt_continue, 'adapter_model.bin')):
            model = AutoModelForCausalLM.from_pretrained(
                train_config.ckpt_continue,
                use_cache=use_cache,
            )
            model = resize_model_embeddings(model, tokenizer, model_family)
            
            peft_config = generate_peft_config(train_config, kwargs)
            model = get_peft_model(model, peft_config)
            current_epoch = int(train_config.ckpt_continue.split('/')[-1].split('-')[-1])
        # 둘다 없을 경우 (아마 새로 학습 하는 경우) --> 여기 걸림림
        else:
            peft_config = generate_peft_config(train_config, kwargs)
            model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    # for name, param in model.named_parameters():
    #     print(f"Parameter name: {name}, Type: {param.dtype}")
    #setting up FSDP if enable_fsdp is enabled

    # 여기 걸림
    if train_config.enable_fsdp:
        # 안걸림 False, False
        if not train_config.use_peft and train_config.freeze_layers:

            freeze_transformer_layers(train_config.num_freeze_layers)

        # 여기 걸림
        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)
        print("to fsdp")
        # use_orig_params=True
        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            # 안걸림
            if train_config.low_cpu_fsdp and rank != 0 else None
        )
        # 위에서 None 값 넣어줌 안걸림
        if ref_model:
            ref_model = FSDP(ref_model, device_id=torch.cuda.current_device())
        # 여기 걸림
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        model.to("cuda")

    
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    lr = train_config.lr
    if current_epoch > 0:
        for cnt in range(current_epoch):
            lr = lr * train_config.gamma
    # Initialize the optimizer and learning rate scheduler
    # 안걸림
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay
        )
    # 걸림
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=train_config.weight_decay
        )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
        current_epoch,
        dataset_config,
        inference_config,
        train_cfg_ins,
        inference_cfg_ins,
        ref_model
    )
    if not train_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

if __name__ == "__main__":
    fire.Fire(main)
