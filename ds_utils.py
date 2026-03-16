import json
import os
import shutil
from pathlib import Path
from typing import Optional

import torch
from deepspeed import zero
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from loguru import logger

import format as fmt


def is_deepspeed_enabled(train_cfg):
    if train_cfg.distributed_type != "deepspeed":
        return False
    
    if train_cfg.deepspeed is None or train_cfg.deepspeed == "":
        raise ValueError("DeepSpeed is enabled but the `deepspeed` configuration path is missing.")
    
    return True

def load_deepspeed_config(train_cfg):
    if not is_deepspeed_enabled(train_cfg):
        raise ValueError("DeepSpeed is not enabled.")

    ds_cfg_path = Path(train_cfg.deepspeed)
    if not ds_cfg_path.exists():
        raise FileNotFoundError(f"DeepSpeed config file not found: {ds_cfg_path}")

    with ds_cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def is_deepspeed_zero3(train_cfg) -> bool:
    ds_cfg = load_deepspeed_config(train_cfg)
    return int(ds_cfg.get("zero_optimization", {}).get("stage", 0)) == 3
    
def gather_16bit_weights_on_model_save_enabled(train_cfg) -> bool:
    # `stage3_gather_16bit_weights_on_model_save` (default: False)
    # For ZeRO stage 3 the state_dict contains just placeholders since the model weights are partitioned across GPUs. 
    # By setting `stage3_gather_16bit_weights_on_model_save=true`,
    # DeepSpeed consolidates the partitioned weights across GPUs and saves them as fp16/bf16.
    # However, if you plan to resume training with ZeRO-3, this option should remain False.
    #
    # See details: 
    # https://deepspeed.readthedocs.io/en/latest/zero3.html
    # https://github.com/huggingface/transformers/issues/28921#issuecomment-1933973037
    if not is_deepspeed_zero3(train_cfg):
        return False
    
    ds_cfg = load_deepspeed_config(train_cfg)
    zero_optimization = ds_cfg["zero_optimization"]
    return zero_optimization.get("stage3_gather_16bit_weights_on_model_save", False)

def save_zero3_resume_checkpoint_enabled(train_cfg) -> bool:
    return train_cfg.save_zero3_resume_checkpoint

def zero3_checkpoint_export_dtype(train_cfg, model_load_kwargs) -> Optional[torch.dtype]:
    if train_cfg.save_zero3_resume_checkpoint:
        return None
    
    mixed_precision = train_cfg.mixed_precision
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    
    return model_load_kwargs["dtype"]

def is_zero3_fp32_checkpoint_export(train_cfg, model_load_kwargs):
    return not train_cfg.use_peft and \
        zero3_checkpoint_export_dtype(train_cfg, model_load_kwargs) == torch.float32

def is_zero3_peft_fp32_export(train_cfg, model_load_kwargs) -> bool:
    return (train_cfg.use_peft and \
        zero3_checkpoint_export_dtype(train_cfg, model_load_kwargs) == torch.float32)

def get_deepspeed_engine(trainer) -> DeepSpeedEngine:
    engine = getattr(trainer, "model_wrapped", None)
    if engine is None:
        raise RuntimeError("Trainer has no `model_wrapped`. DeepSpeed engine was not found.")

    return engine

def save_zero3_resume_checkpoint(trainer, checkpoint_dir):
    engine = get_deepspeed_engine(trainer)
    save_zero3_sharded_checkpoint(engine, checkpoint_dir)

def save_zero3_sharded_checkpoint(engine, checkpoint_dir) -> None:
    """
    Save a ZeRO-3 partitioned DeepSpeed checkpoint.

    The checkpoint is stored in DeepSpeed's distributed format, where
    model parameters are saved as partitioned shards instead of a
    single full gathered model.

    This checkpoint is intended for resuming ZeRO-3 training, 
    not for out-of-the-box inference.

    If a full gathered model is required later for some reasons such as
    inference, it can be reconstructed offline using DeepSpeed's `zero_to_fp32.py`.
    """
    engine.save_checkpoint(checkpoint_dir)

def save_zero3_full_gathered_model(
    trainer,
    train_cfg,
    model_load_kwargs,
    output_dir,
    saved_artifacts,
    dist,
    safe_serialization=True,
):
    export_dtype = zero3_checkpoint_export_dtype(train_cfg, model_load_kwargs)

    # case 1) fp32 export: save ZeRO shards first, then reconstruct fp32 offline/in-process
    if is_zero3_fp32_checkpoint_export(train_cfg, model_load_kwargs):
        checkpoint_dir = os.path.join(output_dir, "zero3_fp32_checkpoint")
        engine = get_deepspeed_engine(trainer)

        logger.info(f"Saving ZeRO-3 sharded checkpoint for fp32 reconstruction to '{checkpoint_dir}'.")
        save_zero3_sharded_checkpoint(engine, checkpoint_dir)

        if dist.is_main_process:
            logger.info(f"Converting ZeRO-3 checkpoint to consolidated fp32 state_dict in '{output_dir}'.")
            convert_zero_checkpoint_to_fp32_state_dict(
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
                safe_serialization=False,
            )
            saved_artifacts.append(f"Float32 full gathered model: {output_dir}")
            logger.info(f"Removing temporary ZeRO-3 checkpoint directory '{checkpoint_dir}'.")
            shutil.rmtree(checkpoint_dir)
        return

    # case 2) bf16/fp16 export: 16-bit gathered save
    accelerator = trainer.accelerator
    engine = get_deepspeed_engine(trainer)
    unwrapped = accelerator.unwrap_model(trainer.model)
    
    state_dict = accelerator.get_state_dict(engine)
    
    if dist.is_main_process:
        unwrapped.save_pretrained(
            output_dir,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=safe_serialization,
        )
        logger.info(f"Saving ZeRO-3 full gathered {fmt.dtype(export_dtype)} model to '{output_dir}'.")
        saved_artifacts.append(f"{fmt.dtype(export_dtype).capitalize()} full gathered model: {output_dir}")

def validate_deepspeed_zero3_config(train_cfg, model_load_kwargs) -> None:    
    if train_cfg.save_zero3_resume_checkpoint and train_cfg.use_peft:
        return

    # full gathered fp32
    export_dtype = zero3_checkpoint_export_dtype(train_cfg, model_load_kwargs)
    if is_zero3_fp32_checkpoint_export(train_cfg, model_load_kwargs):
        if gather_16bit_weights_on_model_save_enabled(train_cfg):
            raise ValueError(
                "For ZeRO-3 fp32 full gathered checkpoint export, "
                "`stage3_gather_16bit_weights_on_model_save` must be false."
            )
    
    # full gathered fp32
    if export_dtype in (torch.float16, torch.bfloat16):
        if not gather_16bit_weights_on_model_save_enabled(train_cfg):
            raise ValueError(
                "For ZeRO-3 fp16/bf16 full gathered checkpoint export, "
                "`stage3_gather_16bit_weights_on_model_save` must be true."
            )

def get_zero3_model_vocab_size(model, dist):
    emb = model.get_input_embeddings().weight

    if hasattr(emb, "ds_id"):
        with zero.GatheredParameters([emb], modifier_rank=0):
            if dist.is_main_process:
                return emb.shape[0]
    else:
        return emb.shape[0]
