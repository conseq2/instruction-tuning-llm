import os
import sys
from contextlib import contextmanager

import bitsandbytes as bnb
import torch
from loguru import logger
from trl.trainer import sft_trainer

import format as fmt

_DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

def setup_logging(dist=None):
    logger.remove()

    is_main_proc = True if dist is None else dist.is_main_process

    logger.add(
        sys.stdout,
        level="INFO",
        colorize=False,
        enqueue=True,
        filter=lambda log: is_main_proc,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level:<8} | "
            f"rank{dist.rank} | "
            "{name}:{function}:{line} - "
            "{message}"
        ),
    )

def build_dataset_log(
    *,
    data_cfg,
    num_total,
    num_after,
    num_train,
    num_valid,
):
    pct_after = (num_after / num_total * 100.0) if num_total else 0.0

    items = [
        ("dataset_path", data_cfg.data_path),
        ("raw", fmt.int_(num_total)),
        ("preprocess", f"{fmt.int_(num_after)} ({fmt.pct(pct_after)})"),
        ("split", f"num_train={fmt.int_(num_train)}, num_valid={fmt.int_(num_valid)}"),
        ("config",f"valid_ratio={data_cfg.valid_ratio}, shuffle={data_cfg.shuffle}, seed={data_cfg.seed}"),
    ]

    return fmt.box("DATASET BUILD", fmt.kv(items))

def find_all_linear_names(model, train_cfg):
    cls = torch.nn.Linear
    if train_cfg.qlora:
        cls = bnb.nn.Linear4bit
    elif train_cfg.llmint8:
        cls = bnb.nn.Linear8bitLt

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            lora_module_names.add(name.split('.')[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')

    return list(lora_module_names)

def count_trainable_params(model):
    total = 0
    trainable = 0

    for p in model.parameters():
        n = getattr(p, "ds_numel", p.numel())
        total += n
        if p.requires_grad:
            trainable += n

    pct = (trainable / total * 100.0) if total > 0 else 0.0
    return trainable, total, pct

def world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def gpu_arch_from_cc(major: int, minor: int) -> str:
    # Ada Lovelace
    if major == 8 and minor >= 9:
        return "Ada"  # e.g. RTX 40

    ARCH_BY_MAJOR = {
        9: "Hopper",          # H100
        8: "Ampere",          # A100, RTX 30
        7: "Volta/Turing",    # V100, T4
        6: "Pascal",          # P100
    }

    return ARCH_BY_MAJOR.get(major, f"SM{major}.{minor}")

def gpu_info():
    if not torch.cuda.is_available():
        return {
            "device": "cpu",
            "name": None,
            "architecture": None,
            "total_memory": None,
        }

    dev = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(dev)

    return {
        "device": f"cuda:{dev}",
        "name": prop.name,
        "architecture": gpu_arch_from_cc(prop.major, prop.minor),
        "total_memory": int(prop.total_memory),
    }

def build_training_logs(
    *,
    train_cfg,
    trainer,
    train_ds,
    valid_ds,
):
    model = trainer.model
    training_args = trainer.args
    # batch
    ws = world_size()
    micro_bs = training_args.per_device_train_batch_size
    grad_accum = training_args.gradient_accumulation_steps
    global_bs = micro_bs * grad_accum * ws
    
    # dataset
    num_train = len(train_ds) if train_ds is not None else 0
    num_valid = len(valid_ds) if valid_ds is not None else 0
    cols = getattr(train_ds, "column_names", None) if train_ds is not None else None

    # trainable params
    trainable, total, pct = count_trainable_params(model)

    # gpu
    g_info = gpu_info()
    
    # mixed precision
    _dtype = {**_DTYPE, "no": "no"}
    mixed_precision = mixed_precision = fmt.dtype(
        _dtype[trainer.accelerator.mixed_precision]
        if trainer.accelerator.mixed_precision is not None
        else None
    )
    model_items = [("mixed_precision", mixed_precision)]
    
    # base model dtype
    quantization_config = getattr(model.config, "quantization_config", None)
    is_4bit = quantization_config and getattr(quantization_config, "load_in_4bit", False)
    is_8bit = quantization_config and getattr(quantization_config, "load_in_8bit", False)
    if is_4bit:
        model_items += [
            ("base_storage_dtype", f"{quantization_config.bnb_4bit_quant_type}"),
            ("base_compute_dtype", fmt.dtype(quantization_config.bnb_4bit_compute_dtype)),
        ]
    elif is_8bit:
        model_items += [
            ("base_storage_dtype", fmt.dtype(torch.int8)),
            ("base_compute_dtype", "int8 + fp16(outlier)"),
        ]
    else:
        base_model = (
            trainer.model.get_base_model()
            if hasattr(trainer.model, "get_base_model")
            else trainer.model
        )
        model_items.append(("base_model_dtype", fmt.dtype(next(base_model.parameters()).dtype)))

    # peft
    use_peft = hasattr(model, "peft_config")
    if use_peft:
        peft_type = "qlora" if is_4bit else "llm.int8" if is_8bit else "lora"
        model_items.append(("peft", peft_type))
        
        model_items.append(("--- adapter ---", ""))
        adapter_dtype = next((fmt.dtype(p.dtype) for p in model.parameters() if p.requires_grad), None)
        
        
        active_adapter = getattr(model, "active_adapter", None)
        peft_config = model.peft_config[active_adapter] if active_adapter else None
        
        tm = peft_config.target_modules
        if tm:
            tm = list(tm)
            target_modules = "\n".join(
                ", ".join(tm[i:i + 3])
                for i in range(0, len(tm), 3)
            )
        else:
            target_modules = None

        ms = peft_config.modules_to_save
        if ms:
            ms = list(ms)
            modules_to_save = "\n".join(
                ", ".join(ms[i:i + 3])
                for i in range(0, len(ms), 3)
            )
        else:
            modules_to_save = None

        model_items += [
            ("adapter_dtype", adapter_dtype),
            ("adapter.r_dim", peft_config.r),
            ("adapter.alpha", peft_config.lora_alpha),
            ("adapter.dropout", peft_config.lora_dropout),
            ("adapter.target_modules", target_modules),
            ("adapter.modules_to_save", modules_to_save),
        ]
    else:
        model_items.append(("peft", False))
    
    objective_items = [
        ("assistant_only_loss", training_args.assistant_only_loss),
        ("max_length", training_args.max_length),
        ("packing", training_args.packing),
    ]
    
    # packing
    if training_args.packing:
        objective_items.append(("packing_strategy", training_args.packing_strategy))
        objective_items.append(("padding_free", training_args.padding_free))
    
    sections = {
        "RUN": [
            ("run", "instruction tuning"),
            ("init_model", train_cfg.init_model),
            ("output_dir", train_cfg.output_dir),
        ],
        
        "DATASET": [
            ("num_train_samples", num_train),
            ("num_valid_samples", num_valid),
            ("columns", cols),
        ],

        "OBJECTIVE": objective_items,
        
        "BATCH": [
            ("world_size", ws),
            ("micro_batch", micro_bs),
            ("accumulation_steps", grad_accum),
            ("global_batch", global_bs),
            ("gradient_checkpointing", training_args.gradient_checkpointing),
        ],
        
        "MODEL": model_items,
        
        "TRAINABLE PARAMS": [
            ("total", fmt.int_(total)),
            ("trainable", fmt.int_(trainable)),
            ("percentage", fmt.pct(pct)),
        ],
        
        "OPTIM & SCHEDULE": [
            ("learning_rate", training_args.learning_rate),
            ("weight_decay", training_args.weight_decay),
            ("max_grad_norm", training_args.max_grad_norm),
            ("warmup_steps", training_args.warmup_steps),
            ("lr_scheduler_type", training_args.lr_scheduler_type),
            ("optim", training_args.optim),
        ],
        
        "EVAL / LOG / SAVE": [
            ("eval_strategy", training_args.eval_strategy),
            ("eval_steps", training_args.eval_steps),
            ("logging_strategy", training_args.logging_strategy),
            ("logging_steps", training_args.logging_steps),
            ("save_strategy", training_args.save_strategy),
            ("save_steps", training_args.save_steps),
            ("save_total_limit", training_args.save_total_limit),
        ],
        
        "SYSTEM": [
            ("seed", training_args.seed),
            ("dataloader_num_workers", training_args.dataloader_num_workers),
        ],
        
        "DEVICE": [
            ("device", g_info["device"]),
            ("gpu_name", g_info["name"]),
            ("architecture", g_info["architecture"]),
            ("total_memory", fmt.gb(g_info["total_memory"]) if g_info["total_memory"] is not None else None),
        ],
    }
    order = [
        "RUN",
        "DATASET",
        "OBJECTIVE",
        "BATCH",
        "MODEL",
        "TRAINABLE PARAMS",
        "OPTIM & SCHEDULE",
        "EVAL / LOG / SAVE",
        "SYSTEM",
        "DEVICE",
    ]
    
    return sections, order


@contextmanager
def capture_lora_init_weights(enabled: bool):
    state = {"lora_init_weights": None}
    if not enabled:
        yield state
        return

    orig_get_peft_model = sft_trainer.get_peft_model

    def wrapped_get_peft_model(model, peft_config, *args, **kwargs):
        peft_model = orig_get_peft_model(model, peft_config, *args, **kwargs)
        if state["lora_init_weights"] is None:
            state["lora_init_weights"] = {
                name: p.detach().float().cpu().clone()
                for name, p in peft_model.named_parameters()
                if p.requires_grad and "lora_" in name
            }
        
        return peft_model

    sft_trainer.get_peft_model = wrapped_get_peft_model
    try:
        yield state
    finally:
        sft_trainer.get_peft_model = orig_get_peft_model
