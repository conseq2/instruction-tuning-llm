import argparse
import json
import os
import shutil

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

def load_meta(meta_path: str) -> dict:
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Merge metadata file not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    required_keys = {
        "base_model",
        "adapter_dir",
        "output_dir",
        "vocab_size",
        "safe_serialization",
        "trust_remote_code",
        "dtype",
    }
    missing = required_keys - set(meta.keys())
    if missing:
        raise ValueError(f"Merge metadata is missing required keys: {missing}")

    return meta

def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in _DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype '{dtype_name}'. "
            f"Expected one of: {sorted(_DTYPE_MAP.keys())}"
        )
    return _DTYPE_MAP[dtype_name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, required=True)
    args = parser.parse_args()

    meta = load_meta(args.meta)

    base_model_id = meta["base_model"]
    adapter_dir = meta["adapter_dir"]
    output_dir = meta["output_dir"]
    vocab_size = meta["vocab_size"]
    safe_serialization = meta["safe_serialization"]
    trust_remote_code = meta["trust_remote_code"]
    dtype = resolve_dtype(meta["dtype"])

    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    os.makedirs(output_dir, exist_ok=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=dtype,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )
    
    orig_vocab_size = base_model.get_input_embeddings().weight.shape[0]
    if orig_vocab_size != vocab_size:
        base_model.resize_token_embeddings(vocab_size)

    merged_model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
    ).merge_and_unload()

    merged_model.save_pretrained(
        output_dir,
        safe_serialization=safe_serialization,
    )

    shutil.rmtree(adapter_dir)
    os.remove(args.meta)

if __name__ == "__main__":
    main()
