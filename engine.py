import json
import os
import shutil
import subprocess
import sys
from typing import Optional

import bitsandbytes as bnb
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from loguru import logger
from omegaconf import OmegaConf
from peft import PeftModel
from transformers import AutoModelForCausalLM
from trl import SFTTrainer

import format as fmt
import utils
from ds_utils import (
    get_deepspeed_engine, 
    get_zero3_model_vocab_size,
    is_deepspeed_zero3, 
    is_zero3_peft_fp32_export,
    save_zero3_full_gathered_model,
    save_zero3_resume_checkpoint,
    validate_deepspeed_zero3_config
)
from exceptions import SanityCheckError
from utils import capture_lora_init_weights


class TrainerEngine:
    def __init__(self, train_cfg, training_args, model, tokenizer, train_ds, model_load_kwargs, dist, valid_ds=None, peft_cfg=None):
        self.train_cfg = train_cfg
        self.training_args = training_args
        self.peft_cfg = peft_cfg
        self.model = model
        self.tokenizer = tokenizer
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.model_load_kwargs = model_load_kwargs
        
        self.dist = dist
        self.trainer = self._build_trainer()
        self.dist.bind_trainer(self.trainer)
        
        self._sanity_check()
        
        if is_deepspeed_zero3(train_cfg):
            validate_deepspeed_zero3_config(train_cfg, model_load_kwargs)
        
    def _build_trainer(self):
        enable_lora_fp32 = getattr(self.train_cfg.peft_config, "enable_lora_fp32", True)
        capture_lora_init = enable_lora_fp32 and (
            getattr(self.model, "is_loaded_in_4bit", False)
            or getattr(self.model, "is_loaded_in_8bit", False)
        )
        
        with capture_lora_init_weights(capture_lora_init) as lora_state:
            trainer = SFTTrainer(
                model=self.model,
                args=self.training_args,
                peft_config=self.peft_cfg,
                train_dataset=self.train_ds,
                eval_dataset=self.valid_ds,
                processing_class=self.tokenizer,
            )
            
        self.trainer = trainer
        if self.peft_cfg:
            self._apply_lora_dtype_policy(lora_state)

        return trainer
    
    def _apply_lora_dtype_policy(self, lora_state=None):
        enable_lora_fp32 = getattr(self.train_cfg.peft_config, "enable_lora_fp32", True)
        
        # In trl 0.29.0, SFTTrainer currently downcasts lora adapter weights to bf16 when the model is loaded in 4bit/8bit.
        if self.adapter_dtype == torch.bfloat16 and enable_lora_fp32:
            if lora_state is None:
                raise RuntimeError("LoRA init weights are required to restore fp32 adapter weights, but lora_state is None.")
            
            init = lora_state["lora_init_weights"]
            if init is None:
                raise RuntimeError("LoRA init weights were not captured but are required to restore fp32 adapters.")
            
            for name, p in self.trainer.model.named_parameters():
                if name in init:
                    p.data = init[name].to(device=p.device, dtype=torch.float32)
                        
        if self.adapter_dtype == torch.float32 and not enable_lora_fp32:
            self.trainer.model.peft_config[self.trainer.model.active_adapter].autocast_adapter_dtype = False
            
            if self.train_cfg.mixed_precision or self.base_model_dtype == torch.float32:
                # When using AMP, trainable weights should never use fp16. 
                # Therefore, only bf16 is currently supported, 
                # and for fp16, enable_lora_fp32 is always set to True.
                #
                # See details: 
                # https://huggingface.co/docs/peft/en/developer_guides/troubleshooting#dtype-related-issues
                target_dtype = torch.bfloat16
            else:
                target_dtype = self.base_model_dtype

            for p in self.trainer.model.parameters():
                if p.requires_grad:
                    p.data = p.data.to(target_dtype)
    
    @property
    def base_model_dtype(self):
        """
        Return the dtype of the base model.
        For quantized PEFT models, this returns the expected dtype of the dequantized base model.
        """
        base_model = (
            self.trainer.model.get_base_model()
            if hasattr(self.trainer.model, "get_base_model")
            else self.trainer.model
        )
        quantization_config = getattr(base_model.config, "quantization_config", None)

        if quantization_config and getattr(quantization_config, "load_in_4bit", False):
            return quantization_config.bnb_4bit_compute_dtype

        if quantization_config and getattr(quantization_config, "load_in_8bit", False):
            return self.model_load_kwargs.get("dtype", torch.float32)

        return next(base_model.parameters()).dtype
    
    @property
    def adapter_dtype(self):
        for p in self.trainer.model.parameters():
            if p.requires_grad:
                return p.dtype
        return None

    def save_model(self, output_dir: Optional[str] = None):
        
        output_dir = output_dir or self.train_cfg.output_dir
        if self.dist.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
        
        saved_artifacts = []
        
        if is_deepspeed_zero3(self.train_cfg):
            self.save_zero3_model(output_dir, saved_artifacts)
        elif self.train_cfg.use_peft:
            self.save_peft_model(output_dir, saved_artifacts)
        else:
            # TODO:
            # `trainer.save_model()` saves `lm_head` and input embeddings as separate tensors
            # without preserving weight tying. For models with tied embeddings, this can break
            # the shared-parameter relationship after reload, causing `lm_head` and embeddings
            # to be updated independently during subsequent training.
            # Therefore, for models that originally use tied embeddings, we should save them
            # in a way that explicitly preserves the tied-weight structure.
            logger.info(f"Saving model to '{output_dir}'.")
            self.trainer.save_model(output_dir)
            saved_artifacts.append(f"Model: {output_dir}")
        
        if self.dist.is_main_process:
            logger.info(f"Saving tokenizer to '{output_dir}'.")
            self.tokenizer.save_pretrained(output_dir)
            saved_artifacts.append(f"Tokenizer: {output_dir}")
            
            cfg_path = os.path.join(output_dir, "train_config.yaml")
            OmegaConf.save(config=self.train_cfg, f=cfg_path)
            saved_artifacts.append(f"Config: {cfg_path}")
            
            logger.info(
                "Save completed.\n"
                "Artifacts:\n"
                + "\n".join(f"  - {item}" for item in saved_artifacts)
            )
        
    def save_peft_model(self, output_dir, saved_artifacts):
        save_with_merge = getattr(self.train_cfg.peft_config, "save_with_merge", False)
        safe_serialization = self.train_cfg.save_safetensors
        is_quantized_peft = self.train_cfg.qlora or self.train_cfg.llmint8
        
        should_load_base_model = (
            not save_with_merge
            or is_quantized_peft
        )

        if should_load_base_model and not is_deepspeed_zero3(self.train_cfg):
            if self.dist.is_main_process:    
                load_msg = "merge with adapters. This is for the 4bit or 8bit lora." if save_with_merge \
                        else "check whether the vocab size has changed. If it has, the resized base model will be saved."
                logger.info(f"Start re-loading the base model on cpu to {load_msg}.")
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.train_cfg.init_model,
                    torch_dtype=self.base_model_dtype,
                    device_map="cpu",
                    trust_remote_code=self.model_load_kwargs.get("trust_remote_code", False),
                )

                orig_vocab = base_model.get_input_embeddings().weight.shape[0]
                cur_vocab = self.trainer.model.get_input_embeddings().weight.shape[0]
                has_vocab_changed = cur_vocab != orig_vocab
                
                if has_vocab_changed:
                    logger.info(
                        f"Embedding size has changed from {orig_vocab} to {cur_vocab}. "
                        f"Resizing the base model embeddings."
                    )
                    base_model.resize_token_embeddings(cur_vocab)
        
        if save_with_merge:
            logger.info("Merging adapter into base model.")
            if is_deepspeed_zero3(self.train_cfg):
                tmp_adapter_dir = os.path.join(output_dir, "adapter")
                self.save_adapter_checkpoint(tmp_adapter_dir, safe_serialization)
                
                unwrapped = self.dist.unwrap_model(self.trainer.model)
                vocab_size = get_zero3_model_vocab_size(unwrapped, self.dist)
                
                if self.dist.is_main_process:
                    merge_meta_path = os.path.join(output_dir, "merge_meta.json")
                    with open(merge_meta_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "base_model": self.train_cfg.init_model,
                                "adapter_dir": tmp_adapter_dir,
                                "output_dir": output_dir,
                                "vocab_size": vocab_size,
                                "safe_serialization": safe_serialization,
                                "trust_remote_code": self.model_load_kwargs.get("trust_remote_code", False),
                                "dtype": str(self.base_model_dtype).replace("torch.", ""),
                            },
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    # In DeepSpeed ZeRO-3, the model cannot be loaded directly on the cpu,
                    # so the model must be loaded in a separate subprocess.
                    subprocess.run(
                        [sys.executable, "merge_zero3_peft_adapter.py", "--meta", merge_meta_path],
                        check=True,
                    )
                    saved_artifacts.append(f"Merged model: {output_dir}")
                return

            # Save the peft model, optionally merging it with the base model.
            # If trained in 4bit or 8bit, the base model should be loaded without quantization.
            if is_quantized_peft:
                tmp_adapter_dir = os.path.join(output_dir, "adapter")
                self.save_adapter_checkpoint(tmp_adapter_dir, safe_serialization)

                if self.dist.is_main_process:
                    merged_model = PeftModel.from_pretrained(
                        base_model, tmp_adapter_dir
                    ).merge_and_unload()

                    shutil.rmtree(tmp_adapter_dir)
            else: # pure lora
                if is_deepspeed_zero3(self.train_cfg):
                    tmp_adapter_dir = os.path.join(output_dir, "adapter")
                    self.save_adapter_checkpoint(tmp_adapter_dir, safe_serialization)

                    if self.dist.is_main_process:
                        merged_model = PeftModel.from_pretrained(
                            base_model,
                            tmp_adapter_dir,
                        ).merge_and_unload()

                        shutil.rmtree(tmp_adapter_dir)             
                else:
                    merged_model = self.trainer.model.merge_and_unload()
            if self.dist.is_main_process:
                logger.info(f"Saving merged model to '{output_dir}'.")
                merged_model.save_pretrained(
                    output_dir, 
                    safe_serialization=safe_serialization
                )
                saved_artifacts.append(f"Merged model: {output_dir}")
            
        else:
            adapter_dir = os.path.join(output_dir, "adapter")
            self.save_adapter_checkpoint(adapter_dir, safe_serialization)
            saved_artifacts.append(f"Adapter: {adapter_dir}")
            
            if self.dist.is_main_process:
                if not is_deepspeed_zero3(self.train_cfg) and has_vocab_changed:
                    logger.info(f"Saving base model to '{output_dir}' since vocab size has changed.")
                    base_model.save_pretrained(
                        output_dir, 
                        safe_serialization=safe_serialization
                    )
                    saved_artifacts.append(f"Base model: {output_dir}")
                
                if is_deepspeed_zero3(self.train_cfg):
                    logger.warning(
                        "Under DeepSpeed ZeRO-3, if the embedding size was changed during training, "
                        "the base model embeddings must be resized manually when merging or loading the adapter."
                    )

    def save_zero3_model(self, output_dir, saved_artifacts):
        if self.train_cfg.save_zero3_resume_checkpoint:
            # this is works for peft model as well
            checkpoint_dir = os.path.join(output_dir, "zero3_resume_checkpoint")
            if self.dist.is_main_process:
                os.makedirs(checkpoint_dir, exist_ok=True)
                
            logger.info(f"Saving ZeRO-3 resume checkpoint to '{checkpoint_dir}'.")
            save_zero3_resume_checkpoint(self.trainer, checkpoint_dir)
            
            if self.dist.is_main_process:
                state_path = os.path.join(checkpoint_dir, "trainer_state.json")
                self.trainer.state.save_to_json(state_path)
            
            saved_artifacts.append(f"ZeRO-3 resume checkpoint: {checkpoint_dir}")
            return
        
        if self.train_cfg.use_peft:
            self.save_peft_model(output_dir, saved_artifacts)
            return
        
        save_zero3_full_gathered_model(
            trainer=self.trainer,
            train_cfg=self.train_cfg,
            model_load_kwargs=self.model_load_kwargs,
            output_dir=output_dir,
            saved_artifacts=saved_artifacts,
            dist=self.dist,
            safe_serialization=self.train_cfg.save_safetensors,
        )
            
    def save_adapter_checkpoint(self, adapter_dir, safe_serialization=True):
        if is_deepspeed_zero3(self.train_cfg) or self.train_cfg.distributed_type=="fsdp":
            unwrapped = self.dist.unwrap_model(self.trainer.model) # PeftModel
            
            if (
                self.adapter_dtype == torch.float32 and
                is_zero3_peft_fp32_export(self.train_cfg, self.model_load_kwargs)
            ):
                checkpoint_dir = os.path.join(os.path.dirname(adapter_dir), "zero3_fp32_checkpoint")
                if self.dist.is_main_process:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                
                logger.info(f"Saving ZeRO-3 sharded checkpoint for fp32 adapter extraction to '{checkpoint_dir}'.")
                save_zero3_resume_checkpoint(self.trainer, checkpoint_dir)
            
                if self.dist.is_main_process:
                    fp32_state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)
                    
                    os.makedirs(adapter_dir, exist_ok=True)
                    unwrapped.save_pretrained(
                        adapter_dir,
                        state_dict=fp32_state_dict,
                        safe_serialization=safe_serialization,
                    )
                
                if self.dist.is_main_process:
                    logger.info(f"Removing temporary ZeRO-3 checkpoint directory '{checkpoint_dir}'.")
                    shutil.rmtree(checkpoint_dir)
            else:    
                # `get_state_dict` consolidates sharded model weights (e.g., fsdp or zero3) into a full state_dict on CPU.
                # While adapter-only gathering is possible, it has led to incomplete or incorrect
                # adapter checkpoints in some sharded setups, so we rely on the full state_dict for correctness
                # such as: https://github.com/huggingface/peft/issues/453
                engine = get_deepspeed_engine(self.trainer)
                full_state_dict = self.trainer.accelerator.get_state_dict(engine)
                if self.dist.is_main_process:
                    os.makedirs(adapter_dir, exist_ok=True)
                    unwrapped.save_pretrained(
                        adapter_dir,
                        state_dict=full_state_dict,
                        safe_serialization=safe_serialization,
                    )
        else:
            os.makedirs(adapter_dir, exist_ok=True)
            self.trainer.model.save_pretrained(
                adapter_dir,
                safe_serialization=safe_serialization,
            )
            
    def _sanity_check(self):
        model = self.trainer.model
        quantization_config = getattr(model.config, "quantization_config", None)
        _dtype = {**utils._DTYPE, "no": "no"}
        
        errors = []
        
        # mixed precision
        runtime_mp = _dtype[self.trainer.accelerator.mixed_precision or "no"]
        expected_mp = _dtype[self.train_cfg.mixed_precision or "no"]
        
        checks = [
            ("mixed precision", runtime_mp, expected_mp),
        ]

        # qlora / llmint8
        if self.train_cfg.qlora or self.train_cfg.llmint8:
            
            # quantized modules
            cls = bnb.nn.Linear4bit if self.train_cfg.qlora else bnb.nn.Linear8bitLt
            has_quant_linear = any(isinstance(module, cls) for module in model.modules())
            
            if not has_quant_linear:
                mode = "qlora" if self.train_cfg.qlora else "llm.int8"
                errors.append(
                    f"{mode} is enabled, but there is no matched bitsandbytes quantized linear modules were found in the model."
                )
                
            # qlora base model compute dtype
            runtime_base_model_compute_dtype = (
                getattr(quantization_config, "bnb_4bit_compute_dtype", None)
                if quantization_config is not None and self.train_cfg.qlora
                else None
            )
            expected_base_model_compute_dtype = self.model_load_kwargs.get("dtype")
            if runtime_base_model_compute_dtype is not None:
                checks.append(
                    (
                        "base model compute dtype",
                        runtime_base_model_compute_dtype,
                        expected_base_model_compute_dtype,
                    )
                )
        
        # adapter dtype
        if self.train_cfg.use_peft:
            enable_lora_fp32 = getattr(self.train_cfg.peft_config, "enable_lora_fp32", True)
            runtime_adapter_dtype = self.adapter_dtype
            if enable_lora_fp32:
                expected_adapter_dtype = torch.float32
            else:
                if self.train_cfg.mixed_precision or self.base_model_dtype == torch.float32:
                    expected_adapter_dtype = torch.bfloat16
                else:
                    expected_adapter_dtype = self.base_model_dtype
            checks.append(("adapter dtype", runtime_adapter_dtype, expected_adapter_dtype))

        for name, actual, expected in checks:
            if actual != expected:
                errors.append(f"{name} is {actual}, but {expected} was expected.")

        if errors:
            raise SanityCheckError(
                "\n" + "\n".join(f"  - {msg}" for msg in errors)
            )

    def _log_train_summary(self):        
        sections, order = utils.build_training_logs(
            train_cfg=self.train_cfg,
            trainer=self.trainer,
            train_ds=self.train_ds,
            valid_ds=self.valid_ds,
        )
        inner = fmt.render(sections, order=order)
        outer = fmt.box("TRAINING SUMMARY", inner.splitlines())
        logger.info(f"\n{outer}")
        
    def run(self):
        self._log_train_summary()
        self.train()
        
        with self.dist.synchronized():            
            if self.train_cfg.distributed_type == "fsdp":
                raise NotImplementedError(
                    "FSDP support is planned for future releases. "
                    "Please use `deepspeed` for now."
                )
                
            if is_deepspeed_zero3(self.train_cfg): # or self.train_cfg.distributed_type=="fsdp":
                self.save_model()
            else:
                if self.dist.is_main_process:
                    self.save_model()

    def train(self):
        self.trainer.train()
