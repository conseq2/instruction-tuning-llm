# LLM Instruction Tuning

This repository provides a framework for instruction tuning Large Language Models. It is designed to handle multi-GPU training environments and efficient fine-tuning using parameter-efficient methods.

## Features

*   **Distributed Training**: Support for training via DeepSpeed (ZeRO stages 0, 1, 2, and 3).
*   **PEFT Support**: Supports LoRA, QLoRA (4-bit), and LLM.int8 (8-bit) fine-tuning.
*   **Assistant-only Loss**: Calculates training loss only on assistant responses by masking user prompts and system instructions based on chat templates.
    *   Requirement: This feature requires a Jinja2 chat template containing `{% generation %}` and `{% endgeneration %}` tags.
    *   Reference: Please refer to the "Train on assistant messages only" section in the [Hugging Face TRL Documentation](https://huggingface.co/docs/trl/sft_trainer#train-on-assistant-messages-only).
*   **Data Packing**: Supports Best-Fit Decreasing (BFD) and Wrapped strategies to pack multiple samples into a single sequence length to maximize training throughput.

## Technical Implementation Details

### 1. Precision and Dtype Policy
To ensure numerical stability during mixed-precision training (especially with QLoRA/LLM.int8), the framework implements a specific precision policy:
*   **FP32 Adapter Weights**: When using quantized base models (4-bit/8-bit), the framework automatically captures and restores LoRA adapter weights in FP32. This prevents "unscaling errors" often encountered in FP16/BF16 mixed-precision setups.

### 2. Data Pipeline and Validation
The `DataPipeline` class performs strict validation on the input dataset:
*   **Role Alternation**: Ensures dialogues strictly follow the `User -> Assistant` or `System -> User -> Assistant` pattern.
*   **Truncation Logic**: Implements prompt-aware truncation. If a sequence exceeds `max_length`, it reduces the assistant's response first to preserve the context of the user's instructions.
*   **Pre-conversion**: Automatically converts simple list-style JSONL files into the standard `{"messages": [...]}` dictionary format.

### 3. Distributed Synchronization and Initialization
*   **Race Condition Prevention**: Uses `local_main_process_first()` context managers during data tokenization and preprocessing to prevent multiple processes from simultaneously writing to the dataset cache.
*   **Initialization Order**: Strictly initializes `TrainingArguments` before model instantiation to ensure DeepSpeed ZeRO-3 or FSDP engine hooks are properly applied during the model loading phase.

### 4. DeepSpeed ZeRO-3 Handling
The framework includes specialized utilities for DeepSpeed ZeRO-3, where model weights are partitioned across all available GPUs:
*   **Automated Weight Gathering**: On model save, the framework collects partitioned shards from all ranks to create a single consolidated checkpoint.
*   **LoRA Merging**: For PEFT training, the framework handles merging the trained adapters back into the base model during the save process.

## Future Work

*   **FSDP Support**: Full support for Fully Sharded Data Parallel (FSDP).
*   **Multi-node Support**: Scaling beyond a single machine.
*   **Preference Tuning**: Support for DPO (Direct Preference Optimization) and ORPO.
*   **RLHF**: Reinforcement Learning from Human Feedback.

## Project Structure

```text
.
├── chat_templates/      # Jinja templates for assistant-only masking
├── configs/             # YAML/JSON configs for Accelerate, DeepSpeed, Models, and Data
├── data/                # Sample dialogues and dataset loading scripts
├── main.py              # Entry point for training
├── engine.py            # Trainer engine wrapping SFTTrainer
├── model.py             # Model and Tokenizer factories
├── data.py              # Data pipeline and preprocessing logic
├── distributed.py       # Distributed environment helpers
├── ds_utils.py          # DeepSpeed ZeRO-3 specific utilities
├── utils.py             # Logging and parameter utilities
└── scripts/             # Shell scripts for launching training
```

## Dataset Format

The pipeline expects JSONL format. Refer to `data/sample_dialogue/` for examples.

### Single-turn Example
```json
{"messages": [{"role": "user", "content": "Tell me a joke."}, {"role": "assistant", "content": "Why did the chicken cross the road? To get to the other side."}]}
```

### Multi-turn Example
```json
{"messages": [{"role": "user", "content": "Who is the author of 'Dragon Raja'?"}, {"role": "assistant", "content": "The author is Yeong-do Lee."}, {"role": "user", "content": "What is his other famous work?"}, {"role": "assistant", "content": "He also wrote 'The Bird That Drinks Tears'."}]}
```

## Installation

```bash
pip install -r requirements.txt
```
Note: Flash Attention (v2 or v3) is not included in `requirements.txt`. Please install the version compatible with your specific CUDA environment manually.

## Usage

Training is executed through provided shell scripts. Before running, adjust the configuration files in the `configs/` directory to match your hardware and preferences.

### 1. Configuration
Modify the following files based on your environment:
*   **Accelerate Config**: Files in `configs/accelerate/` (e.g., number of GPUs, mixed precision).
*   **DeepSpeed Config**: Files in `configs/deepspeed/` (e.g., ZeRO stage, offloading).
*   **Training Config**: `configs/train/train.yaml` (e.g., hyperparameters, PEFT settings).

### 2. Execution

#### Single-GPU Training
For single-GPU setups, it is recommended to use DeepSpeed Stage 0. 
1. Set `distributed_type: deepspeed` in your training config.
2. Set the `deepspeed` path to `configs/deepspeed/default/ds_stage0.json`.
3. Run the script:
```bash
bash run_single_gpu_train.sh
```

#### Multi-GPU Training
For multi-GPU setups, adjust the `CUDA_VISIBLE_DEVICES` and `ACCELERATE_CFG` in the script to match your preference.
```bash
bash run_multi_gpu_train.sh
```
