import argparse

from config import Configurator
from data import DataPipeline
from distributed import Dist
from engine import TrainerEngine
from model import ModelFactory
from utils import setup_logging


def main():
    
    # distributed helper
    dist = Dist()
    
    # logging
    setup_logging(dist)
    
    # arguments
    args = parse_args()
    
    # configurations
    model_cfg = Configurator.load_config(args.model_config)
    data_cfg = Configurator.load_config(args.data_config)
    train_cfg = Configurator.load_config(args.train_config)
    
    train_cfg = Configurator.extend_config(
        train_cfg,
        {"init_model": model_cfg.init_model},
    )

    # build tokenizer
    factory = ModelFactory(model_cfg)
    tokenizer = factory.create_tokenizer(
        assistant_only_loss=train_cfg.assistant_only_loss
    )
    
    # prepare data
    max_seq_len = Configurator.resolve_max_seq_len(train_cfg, model_cfg)
    pipeline = DataPipeline(
        data_cfg=data_cfg,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        packing=train_cfg.packing
    )

    with dist.local_main_process_first():
        # TODO:
        # In multi-node training with a shared filesystem, local main processes from different
        # nodes may enter pipeline.build() concurrently, which can lead to race conditions
        # during cache creation.
        #
        # Therefore, there may be need to add logic to check whether the filesystem 
        # is shared across nodes and branch the synchronization behavior accordingly.
        # i.e., main_process_first() for shared filesystem cache/files, 
        # and local_main_process_first() for node-local files.
        train_ds, valid_ds = pipeline.build()

    if valid_ds is None:
        train_cfg = Configurator.disable_eval(train_cfg)

    # build training arguments
    # If using ZeRO initialization, instantiate your model after initializing
    # `TrainingArguments`, otherwise ZeRO won't be applied.
    #
    # Likewise, When using FSDP `cpu_ram_efficient_loading`, the distributed process group
    # must be initialized before calling `from_pretrained`; with the Trainer API,
    # this happens when `TrainingArguments` is created.
    #
    # Therefore, building training arguments before model loading is required.
    #
    # See details:
    # https://huggingface.co/docs/transformers/main_classes/trainer
    # https://huggingface.co/docs/accelerate/usage_guides/fsdp
    training_args = Configurator.build_sft_training_args(train_cfg)
    
    # build model
    model_load_kwargs = Configurator.build_model_load_kwargs(train_cfg, model_cfg)
    model = factory.create_model(**model_load_kwargs)
    
    # build peft config
    peft_cfg = Configurator.build_peft_config(train_cfg, model) if train_cfg.use_peft else None
    
    # build training engine
    engine = TrainerEngine(
        train_cfg=train_cfg, 
        training_args=training_args, 
        model=model, 
        tokenizer=tokenizer, 
        train_ds=train_ds, 
        valid_ds=valid_ds,
        peft_cfg=peft_cfg,
        model_load_kwargs=model_load_kwargs,
        dist=dist,
    )
    engine.run()

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Fine-tuning Framework")
    parser.add_argument("--model_config", type=str, required=True, help="Path to model config")
    parser.add_argument("--data_config", type=str, required=True, help="Path to data config")
    parser.add_argument("--train_config", type=str, required=True, help="Path to training config")

    return parser.parse_args()

if __name__ == "__main__":
    main()
