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

First, fine-tuning the TinyLlama-1.1B using **full precision** and a **batch size of 32**:

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

#### Result:
* The baseline model could not fit in the 80GB GPU memory. 

* The reduced batch size has a memory usage of 30.41GB but was slower 137.04s. The reason behind is that a smaller batch size reduces the memory footprint, but also reduces the efficiency of parallel computing, so the training becomes slower.

* The gradient accumulation makes the training much faster with 57.19s, 7646.24 tok/s but slightly increased memory usage to 34.82GB. The reason behind is that multiple small batches are accumulated before updating weights, improving computing efficiency without significantly increasing memory usage.
 
* The reduced precision further reduced memory usage to 20.10GB and faster training by using 34.97s, 12506.13 tok/s. Because lower precision reduces memory per parameter and speeds up computations and the consequence is lower model accuracy.

* The mixed precision strategy results in slightly more memory used 31.32GB, and training slightly slower to 39.69s, 11016.98 tok/s. It's because it balances memory savings and numerical stability and lead to a trade-off between speed and accuracy.



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

#### Result:
* Based on the results of the litgpt output, training time increases as model size grows. 1.1B to 3B: Memory increased from 34.82GB to 46.89GB, training time increased from 57s to 83.8s. And 3B to 7B: Memory increased from 46.89GB to 69.71GB, training time increased from 83.8s to 138s. 

* Changing the optimizer helps because Adam has two state values per parameter, consuming about 3 times memory compared to the model itself. While SGD only tracks gradients, lowering memory a lot.


#### Parameter Efficient Fine-Tuning


#### LoRA (Low-Rank Adaptation)

LoRA is a parameter-efficient fine-tuning method that works through the following mechanisms:

1. **Core Principle**: LoRA freezes the pre-trained model weights and injects trainable low-rank matrices into each Transformer layer.
2. **Low-Rank Decomposition**: It represents weight updates as a product of two smaller matrices, significantly reducing the number of parameters that need to be trained and stored.
3. **Mathematical Representation**: If W is the original weight matrix, LoRA represents the update as W + ΔW = W + BA, where B and A are low-rank matrices.
4. **Advantages**: Only a small fraction of parameters need to be trained (typically 0.1%-1% of the original model), saving memory and computational resources.

#### QLoRA (Quantized LoRA)

QLoRA extends LoRA by incorporating quantization techniques:

1. **Quantized Base Model**: QLoRA first quantizes the base model (typically to 4-bit precision), substantially reducing memory footprint.
2. **High-Precision Computation**: While storage uses low precision, high-precision computations are maintained for gradient calculations during backpropagation.
3. **Technical Innovations**:
   - 4-bit NormalFloat (NF4) quantization
   - Double quantization techniques
   - Paged optimizer
4. **Advantages**: Enables fine-tuning of large language models on consumer hardware while maintaining performance.

In TinyLlama applications, these techniques allow for efficient customization of the model even in resource-constrained environments, without having to train or store the full model parameters. These methods are particularly suitable for already lightweight models like TinyLlama, further enhancing their applicability on resource-limited devices.


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

#### Result:
* 1.1B model with LoRA only use a small 8.00 GB memory and has a training time of 45.17s. It's because it fine-tuning only low-rank matrices, not the full model, makes training faster and more efficient, requiring minimal memory.

* 3B model with LoRA has 86.35s training time and 17.02 GB memory usage. It's because as the model size increases, both training time and memory usage grow, but still remain relatively low due to the parameter-efficient fine-tuning feature of LoRA.

* 7B model with LoRA has longer training time 133.42s and bigger memory usage 29.99 GB. The reason is same as the increase from 1.1B to 3B. Larger model longer time.

* 7B model with LoRA and Quantization technique has longer time of 182.81s, but less memory usage of 21.25 GB. The reason is that quantization can further reduces memory usage by representing model weights with lower precision, but it will give slower training speed. 

* The final 13B model with LoRA has largest 222.68s training time and 51.46 GB memory. But LoRA still allows training with reasonable memory usage and relatively fast times.

## Part 2: Multi-GPU Training

This section demonstrates distributed training across multiple GPUs. Two separate tracks are provided - one for systems with 4x A100 80GB GPUs and another for systems with 4x V100 32GB GPUs.

### Distributed Data Parallel (DDP)

DDP is a model parallel training approach that enables efficient distributed training across multiple GPUs or machines.

#### How DDP Works:
1. **Data Parallelism**: Each GPU maintains a complete copy of the model
2. **Gradient Synchronization**: During backpropagation, gradients from all replicas are synchronized (typically using all-reduce operations)
3. **Identical Updates**: All model replicas receive identical parameter updates

#### Advantages:
- Relatively simple implementation 
- Near-linear scaling with number of GPUs for many workloads
- Well-established technique with mature implementations in frameworks like PyTorch

#### Limitations:
- Memory requirements scale poorly as model size increases
- Each GPU must store the entire model, parameters, gradients, and optimizer states

### Fully Sharded Data Parallel (FSDP)

FSDP is an advanced distributed training technique that addresses the memory limitations of DDP.

#### How FSDP Works:
1. **Model Sharding**: The model parameters, gradients, and optimizer states are sharded across GPUs
2. **Dynamic Communication**: During forward and backward passes, the required parameters are gathered from other GPUs as needed
3. **Compute-Communication Overlap**: Optimized to overlap computation with communication

#### Advantages:
- **Memory Efficiency**: Dramatically reduces per-GPU memory requirements
- **Scalability**: Enables training of much larger models than possible with DDP
- **Flexibility**: Configurable sharding strategies to balance memory savings and communication overhead

#### Performance Effects:

Both techniques show different performance characteristics:

| Aspect | DDP | FSDP |
|--------|-----|------|
| Memory Usage | High (full model on each GPU) | Low (sharded across GPUs) |
| Communication Volume | Lower (only gradients) | Higher (parameters, gradients) |
| Scaling Efficiency | Very good up to memory limits | Good even for massive models |
| Implementation Complexity | Lower | Higher |
| Training Speed | Faster for smaller models | Better for large models that wouldn't fit in DDP |



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

#### Single GPU Baseline (OpenLLaMA 7B)(run inside)

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

#### Result:
* Memory Usage on Single GPU:61.97 GB allocated, 71.35 GB reserved
Training Time: 3 minutes 32 seconds for 400 steps.

* Memory Usage on 4x A100 80GB:
Memory Used per GPU: Around 67.83 GB to 73.98 GB allocated per GPU
Training Time: 2 minutes 46 seconds for 100 steps

* DDP does not reduce the total memory required for the training job. The potential benefit for using DDP is that it can reduce training time, as each GPU handles a subset of the data and model, allowing for parallel processing. In this example, training on 4 GPUs took 2 minutes 46 seconds, while training on 1 GPU took 3 minutes 32 seconds. I may not always get a benefit from using DDP because sometimes for smaller models or when the batch size is small, the overhead of synchronizing the GPUs can actually slow down the training process. DDP benefits most when the model size is large enough to be split effectively. 

* Memory Usage with FSDP: Allocated Memory per GPU: 30.18 GB to 36.33 GB, Training Time: 2 minutes 53 seconds for 100 steps
* Memory Usage with FSDP and Larger Batch Size: Allocated Memory per GPU: 35.23 GB to 53.54 GB, Training Time: 1 minute 50 seconds for 50 steps
* FSDP significantly reduces memory usage to about 35 GB compared to DDP. It shard optimizer states, gradients, and parameters across multiple GPUs.
The potential benefit of using FSDP is that it increase memory efficiency, allowing training of larger models that may not fit on a single GPU. Yes, I've observed that the memory savings from about 70 GB with DDP to 35 GB with FSDP. I can always get the benefit of less memory usage but I may not always get a benefit from using FSDP because split the model parameters across GPUs required time and resources, which can probably slow down training for smaller models sometimes.


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
