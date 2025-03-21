# Large-scale Model Training on Cloud 

## Overview

This experiment demonstrates how to fine-tune large language models on Chameleon cloud infrastructure using different optimization techniques. Here are the several approaches:

1. Gradient accumulation
2. Reduced precision (bf16, mixed precision)
3. Parameter efficient fine-tuning (LoRA, QLoRA)
4. Distributed training (across multiple GPUs and with CPU offload)

The experiment is structured in two main sections:
- **Single GPU training**: Working with an A100 80GB GPU to explore batch size impacts, gradient accumulation, precision settings, and parameter-efficient fine tuning
- **Multi-GPU training**: Utilizing 4x A100 80GB or 4x V100 32GB GPUs to implement distributed data parallelism and fully sharded data parallelism

## Models and Tools

### Models Used

#### TinyLlama
TinyLlama is a compact yet powerful language model with 1.1 billion parameters. Introduced in a paper published in 2024 (arXiv:2401.02385), it's designed to be a resource-efficient alternative to larger LLMs. Despite its relatively small size, TinyLlama delivers impressive performance across various NLP tasks. It was trained on approximately 3 trillion tokens, making it suitable for tasks that require deep language understanding while being more accessible for fine-tuning experiments on limited hardware.

#### OpenLLaMA
OpenLLaMA is an open-source reproduction of Meta's LLaMA architecture, developed by the OpenLM Research team. In this experiment, we work with several versions:
- OpenLLaMA 3B: A 3 billion parameter version
- OpenLLaMA 7B: A 7 billion parameter version
- OpenLLaMA 13B: A 13 billion parameter version

These models provide progressively more powerful language capabilities at the cost of increased computational requirements. OpenLLaMA models are trained on diverse datasets and offer a good balance between performance and accessibility for research purposes.

### LitGPT

LitGPT is a toolkit developed by Lightning AI that simplifies working with large language models. It serves as a convenient wrapper around PyTorch Lightning capabilities, providing:

1. **Easy fine-tuning**: Streamlined processes for adapting pre-trained models to specific tasks
2. **Built-in optimization techniques**: Support for gradient accumulation, mixed precision, and more
3. **Recipe-based approach**: Can define training configurations in YAML files
4. **Integration with PEFT methods**: Support for parameter-efficient fine-tuning techniques like LoRA
5. **Distributed training support**: Built-in capabilities for training across multiple GPUs

LitGPT abstracts away much of the complexity involved in setting up training loops, distributed processes, and optimization strategies, allowing researchers to focus on model architecture and performance rather than implementation details.


## Experiment Setup

### Requirements

- Bare metal instance on Chameleon with NVIDIA GPU capabilities
- "Single GPU" section: GPU with NVIDIA CUDA compute capability 8.0+ (A100 or A30)
- "Multiple GPU" section: Node with 4 GPUs (either 4x A100 80GB or 4x V100 32GB)

### Initial Setup

First, we pull the necessary container for the "Multiple GPU" section (this will take some time, so we start it early):

```bash
docker pull pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
```

Install monitoring tools:

```bash
sudo apt update; sudo apt -y install nvtop
```

## Part 1: Single GPU Training (A100 80GB)

Use an A100 80GB GPU to explore different techniques for training large models on a single GPU.

### Setting Up the Environment (all parts run in colab)

First, verify the GPU is available:

```bash
nvidia-smi
```

Install LitGPT:

```bash
pip install 'litgpt[all]'==0.5.7 'lightning<2.5.0.post0'
```

Download foundation models:

```bash
litgpt download TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
litgpt download openlm-research/open_llama_3b
litgpt download openlm-research/open_llama_7b
litgpt download openlm-research/open_llama_13b
```

Get the fine-tuning "recipes":

```bash
git clone https://github.com/xxx
```

### Experiments on Single GPU

#### Baseline Experiment

First, we attempt fine-tuning the TinyLlama-1.1B using full precision and a batch size of 32:

```bash
litgpt finetune_full --config config/tiny-llama-full.yaml --train.global_batch_size 32 --train.micro_batch_size 32
```

This fails because the model doesn't fit in the 80GB of GPU memory.

#### Experiment: Reduced Batch Size

Reducing the batch size allows the model to fit:

```bash
litgpt finetune_full --config config/tiny-llama-full.yaml --train.global_batch_size 8 --train.micro_batch_size 8
```

#### Experiment: Gradient Accumulation

Using gradient accumulation allows us to work with a larger effective batch size:

```bash
litgpt finetune_full --config config/tiny-llama-full.yaml --train.global_batch_size 32 --train.micro_batch_size 8
```

#### Experiment: Reduced Precision

Using brain float16 format reduces memory requirements:

```bash
litgpt finetune_full --config config/tiny-llama-full.yaml --train.global_batch_size 32 --train.micro_batch_size 8 --precision bf16-true
```

#### Experiment: Mixed Precision

Mixed precision trading off some memory for better precision:

```bash
litgpt finetune_full --config config/tiny-llama-full.yaml --train.global_batch_size 32 --train.micro_batch_size 8 --precision bf16-mixed
```

#### Experiment: Training Larger Models

Scaling up to a 3B parameter model:

```bash
litgpt finetune_full --config config/open-llama-3b-full.yaml --train.global_batch_size 32 --train.micro_batch_size 8 --precision bf16-true
```

Scaling up to a 7B parameter model:

```bash
litgpt finetune_full --config config/open-llama-7b-full.yaml --train.global_batch_size 16 --train.micro_batch_size 4 --precision bf16-true
```

Attempting a 13B model (fails with OOM):

```bash
litgpt finetune_full --config config/open-llama-13b-full.yaml --train.global_batch_size 1 --train.micro_batch_size 1 --precision bf16-true
```

Using SGD optimizer for the 13B model (slower but fits in memory):

```bash
litgpt finetune_full --config config/open-llama-13b-full.yaml --train.global_batch_size 1 --train.micro_batch_size 1 --precision bf16-true --optimizer SGD --train.max_steps 25
```

#### Parameter Efficient Fine-Tuning

Using LoRA for the 1.1B model:

```bash
litgpt finetune --config config/tiny-llama-lora.yaml
```

Using LoRA for the 3B model:

```bash
litgpt finetune --config config/open-llama-3b-lora.yaml
```

Using LoRA for the 7B model:

```bash
litgpt finetune --config config/open-llama-7b-lora.yaml
```

Using QLoRA (quantized LoRA) for the 7B model:

```bash
litgpt finetune --config config/open-llama-7b-lora.yaml --quantize bnb.nf4
```

Using LoRA for the 13B model:

```bash
litgpt finetune --config config/open-llama-13b-lora.yaml
```

## Part 2: Multi-GPU Training

This section demonstrates distributed training across multiple GPUs. Two separate tracks are provided - one for systems with 4x A100 80GB GPUs and another for systems with 4x V100 32GB GPUs.

### Setup for Multi-GPU Training

Start by checking and stopping any running containers:

```bash
docker ps
docker stop CONTAINER  # Replace CONTAINER with actual container name/ID if needed
```

Start the PyTorch container with GPU access:

```bash
docker run -it -v /home/cc/llm-chi/torch:/workspace --gpus all --ipc host pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
```

Install required Python libraries inside the container:

```bash
pip install 'litgpt[all]'==0.5.7 'lightning<2.5.0.post0'
```

Download the foundation model:

```bash
litgpt download TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
```

In a second terminal, start the GPU monitoring tool:

```bash
nvtop
```

### A100 GPU Experiments

#### Single GPU Baseline (OpenLLaMA 7B)

```bash
python3 a100_llama7b_1device.py
```

#### Distributed Data Parallelism (DDP) with 4 GPUs

```bash
python3 a100_llama7b_4ddp.py
```

#### Fully Sharded Data Parallelism (FSDP) with 4 GPUs

```bash
python3 a100_llama7b_4fsdp.py
```

#### FSDP with Larger Batch Size

```bash
python3 a100_llama7b_4fsdp_8batch.py
```

#### (Optional) DeepSpeed with CPU Offloading for 13B Model

Install DeepSpeed:

```bash
pip install deepspeed==0.14.0
```

Download the 13B model:

```bash
litgpt download openlm-research/open_llama_13b
```

Run the training:

```bash
python3 a100_llama13b_deepspeed.py
```

### V100 GPU Experiments

#### Single GPU Baseline (TinyLlama 1.1B)

```bash
python3 v100_llama1b_1device.py
```

#### Distributed Data Parallelism (DDP) with 4 GPUs

```bash
python3 v100_llama1b_4ddp.py
```

#### Fully Sharded Data Parallelism (FSDP) with 4 GPUs

```bash
python3 v100_llama1b_4fsdp.py
```

#### FSDP with Larger Model (OpenLLaMA 3B)

```bash
python3 v100_llama3b_4fsdp.py
```

Attempting the same model on a single V100 (will fail with OOM):

```bash
python3 v100_llama3b_1device.py
```

### Debugging

If a training job crashes with OOM (Out of Memory), you can stop all Python processes:

```bash
pkill -9 python
```

## Key Findings

1. **Batch Size Impact**: Smaller batch sizes reduce memory requirements but may slow down training
2. **Gradient Accumulation**: Allows for larger effective batch sizes without significant memory increases
3. **Precision Settings**: Reduced precision (bf16) significantly decreases memory requirements with minimal loss in quality
4. **LoRA and QLoRA**: Dramatically reduce memory requirements for fine-tuning, enabling training of much larger models
5. **Distributed Training**: 
   - DDP increases effective batch size without reducing per-GPU memory
   - FSDP reduces per-GPU memory requirements by sharding parameters, gradients, and optimizer states
   - DeepSpeed with CPU offload enables training extremely large models with limited GPU memory

This experiment highlights the various techniques that can be employed to train large language models efficiently, even with limited GPU resources.
