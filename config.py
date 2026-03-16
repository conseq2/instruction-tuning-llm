from typing import Dict, Optional

import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType
from transformers import AutoConfig, BitsAndBytesConfig, HfArgumentParser
from trl import SFTConfig

import utils

DEFAULT_MAX_LEN = 8192

class Configurator:
    @staticmethod
    def load_config(cfg_path: str) -> DictConfig:
        # TODO: override if needed
        return OmegaConf.load(cfg_path)
    
    @staticmethod
    def resolve_max_seq_len(train_cfg, model_cfg) -> int:
        if train_cfg.max_length:
            return train_cfg.max_length
        
        model_spec = AutoConfig.from_pretrained(model_cfg.init_model)
        train_cfg.max_length = getattr(model_spec, "max_position_embeddings", DEFAULT_MAX_LEN)
        return train_cfg.max_length
            
    @staticmethod
    def build_model_load_kwargs(train_cfg, model_cfg) -> Dict:
        mp = train_cfg.mixed_precision
        load_dtype = model_cfg.load_dtype
        
        if load_dtype == "auto":
            load_dtype = AutoConfig.from_pretrained(model_cfg.init_model).dtype
        else:
            try:
                load_dtype = utils._DTYPE[model_cfg.load_dtype]
            except KeyError:
                raise ValueError(
                    f"Invalid load_dtype: {model_cfg.load_dtype}. "
                    "Choose from: auto | fp32 | bf16 | fp16"
                )
                
        bf16_support = torch.cuda.is_bf16_supported()
        if (load_dtype == torch.bfloat16 or mp == 'bf16') and not bf16_support:
            raise ValueError('Your GPU does not support bfloat16. Check the device support and configure accordingly')
        
        if train_cfg.qlora and train_cfg.llmint8:
            raise ValueError(
                "'qlora' and 'llmint8' can't both be set to True. "
                "If you want to train with quantized PEFT, choose only one of them."
            )
        #TODO: compute_dtye will follow the mixed precision compared to load_dtype (so need to fix it)
        if train_cfg.qlora:
            quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=load_dtype,
                )
        elif train_cfg.llmint8:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = None
        
        kwargs = {
            "dtype": load_dtype,
            "trust_remote_code": getattr(model_cfg, "trust_remote_code", False),
        }
        if quantization_config:
            kwargs["quantization_config"] = quantization_config
        
        # NOTE: Consider allowing FlashAttention in fp32 training.
        # It runs with accelerate only but causes unintended downcasting and is unsupported with deepspeed.
        attn_impl = train_cfg.attn_implementation
        is_flash_attn_unsupported = load_dtype == torch.float32 and mp is None
        if is_flash_attn_unsupported and attn_impl in ("flash_attention_2", "flash_attention_3"):
            raise ValueError("FlashAttention only supports fp16 or bf16 dtypes and cannot be used for fp32 training. "
                             "Switch the attention implementation to sdpa for fp32 training.")
            
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl

        return kwargs
    
    @staticmethod
    def build_sft_training_args(train_cfg) -> SFTConfig:
        if train_cfg.packing and train_cfg.packing_strategy.lower() == "bfd":
            # When packing is enabled with strategy "bfd", padding-free is enabled, regardless of the value of this parameter.
            # See details: https://huggingface.co/docs/trl/en/sft_trainer
            if not train_cfg.padding_free:
                logger.info("padding_free will be set to True to support bfd packing.")
            train_cfg.padding_free = True
        
        if train_cfg.packing and train_cfg.padding_free:
            if train_cfg.attn_implementation not in ("flash_attention_2", "flash_attention_3"):
                # Currently, padding_free is only supported with the FlashAttention 2 or 3, which can efficiently handle the flattened batch structure. 
                # See details: https://huggingface.co/docs/trl/en/sft_trainer
                raise ValueError(
                    "padding_free requires FlashAttention 2 or 3. "
                    "Use 'flash_attention_2' or 'flash_attention_3' for attn_implementation."
                )
        if train_cfg.packing and not train_cfg.padding_free:
            # It’s highly recommended to use padding-free batching with FlashAttention 2 or FlashAttention 3. 
            # Otherwise, you may encounter batch contamination issues.
            # See details: https://huggingface.co/docs/trl/v0.25.0/reducing_memory_usage
            logger.warning(
                "Using packing without padding-free batching may cause batch contamination. "
                "Padding-free batching with FlashAttention 2 or 3 is highly recommended."
            )
        
        # https://github.com/huggingface/transformers/issues/39849
        # & accelerate automatically set these values to True.
        # To avoid unexpected behavior, set these flags before building SFTConfig.
        train_cfg.bf16, train_cfg.bf16_full_eval = False, False
        match train_cfg.mixed_precision:
            case "fp16":
                train_cfg.fp16 = True
                train_cfg.fp16_full_eval = True
            case "bf16":
                train_cfg.bf16 = True
                train_cfg.bf16_full_eval = True

        # TODO: optionally set the wandb run name.
        # Configurator.build_wandb_run_name(train_cfg)
        
        train_cfg = OmegaConf.to_container(train_cfg, resolve=True)
        parser = HfArgumentParser(SFTConfig)
        training_args: SFTConfig = parser.parse_dict(
            train_cfg,
            allow_extra_keys=True
        )[0]
        
        training_args.dataset_text_field = "messages"
        
        return training_args            

    @staticmethod
    def build_peft_config(train_cfg, model) -> Optional[LoraConfig]:
        peft_config = train_cfg.peft_config
        
        if hasattr(peft_config, "target_modules"):
            target_modules = list(peft_config.target_modules)
        else:
            target_modules = utils.find_all_linear_names(model, train_cfg)
        
        modules_to_save = getattr(peft_config, "modules_to_save", None)
        
        # When using AMP, trainable weights should never use fp16.
        # Attempting to use fp16 trainable parameters with fp16 AMP can lead to:
        # "ValueError: Attempting to unscale FP16 gradients".
        #
        # See details: 
        # https://huggingface.co/docs/peft/en/developer_guides/troubleshooting#dtype-related-issues
        if train_cfg.mixed_precision == "fp16" and not train_cfg.peft_config.enable_lora_fp32:
            logger.warning("fp16 amp doesn't work with fp16 adapter weights, so adapter weights will be set to fp32.")
            train_cfg.peft_config.enable_lora_fp32 = True

        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=peft_config.r_dim,
            lora_alpha=peft_config.lora_alpha,
            lora_dropout=peft_config.lora_dropout,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
        )

    @staticmethod
    def extend_config(
        cfg,
        patch,
        *,
        override: bool = True,
        return_dict: bool = False,
    ):
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)

        if not isinstance(patch, DictConfig):
            patch = OmegaConf.create(patch)

        if override:
            merged = OmegaConf.merge(cfg, patch)
        else:
            merged = OmegaConf.merge(patch, cfg)

        if return_dict:
            return Configurator.to_dict(merged)

        return merged
    
    @staticmethod
    def disable_eval(train_cfg):
        return Configurator.extend_config(
            train_cfg,
            {
                "do_eval": False,
                "eval_strategy": "no",
                "eval_steps": None,
            },
            override=True,
        )

    @staticmethod
    def to_dict(cfg) -> Dict:
        if isinstance(cfg, DictConfig):
            return OmegaConf.to_container(cfg, resolve=True)

        if isinstance(cfg, dict):
            return cfg

        raise TypeError(f"'{Configurator.to_dict.__name__}' expected DictConfig or dict as input, got {type(cfg)}")

    @staticmethod
    def update_wandb_run_name(train_cfg):
        # TODO: Re-name wandb `run_name`` based on the train_cfg settings.
        pass
