# Quantized-LoRA-Finetuning-of-GPT-2


## Overview

This repository explores the performance of the GPT-2 model under multiple variants of quantization methods, as an attempt to achieve a more efficient and flexible implementation of large language models (LLMs).

## Methods and References

- Quantization for LLM [(LLM-QAT: Data-Free Quantization Aware Training for Large Language Models)](https://arxiv.org/abs/2305.17888)

- LoRA Fine-tuning [(LoRA: Low-Rank Adaptation of Large Language Models)](https://arxiv.org/abs/2106.09685)

- Switchable-Precision Scheme [(InstantNet: Automated Generation and Deployment of Instantaneously Switchable-Precision Networks)](https://arxiv.org/pdf/2104.10853.pdf)

- Cyclic Precision Training [(CPT: Efficient Deep Neural Network Training via Cyclic Precision)](https://arxiv.org/abs/2101.09868)

- Adversarial Attacks for LLM [(Gradient-based Adversarial Attacks against Text Transformers)](https://arxiv.org/abs/2104.13733)


## Implementation Guidance

### Prerequisites
- See ***env.yml*** for the complete conda environment. Create a new conda environment:
```
conda env create -f env.yml
conda activate pytorch
```


### Attempt 1: Full finetuning of GPT-2 under static precision 
1. Go to the project directory for static precision training:
```
cd static_precision
```

2. Run ***run_qa.py***. This example code fine-tunes GPT-2 on the SQuAD1.0 dataset with 8 bits per quantized value:
```
python run_qa.py --model_name_or_path gpt2 \
--dataset_name squad \
--do_train \
--do_eval \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--seed 42 \
--learning_rate 3e-05 \
--logging_strategy steps \
--logging_steps 10 \
--lr_scheduler_type linear \
--report_to tensorboard \
--gradient_accumulation_steps 1 \
--max_seq_length 384 \
--doc_stride 128 \
--max_steps 1000 \
--w_bits 4 \
--a_bits 4 \
--kv_bits 4 \
--output_dir ./results
```

The layers to be quantized when running this code include:
- The query/key/value projection in each attention block (quantized by **w_bits**)
- The KV cache (quantized by **kv_bits**)
- The output projection in each attention block (quantized by **w_bits**)
- The two linear layers in each MLP module (quantized by **w_bits**)
- The final linear layer (quantized by **w_bits**)
- The inputs for these above layers (quantized by **a_bits**)


### Attempt 2: LoRA finetuning of GPT-2 under switchable precision 
1. Go to the project directory for switchable precision LoRA training:
```
cd switchable_precision
```

2. Run ***run_qa.py***. This example code fine-tunes GPT-2 on the SQuAD1.0 datas with 2 LoRA modules (2 pairs of LoRA adapters with rank 16) per layer, one of which quantized by 32 bits per value and the other quantized by 8 bits per value: 
```
python run_qa.py --model_name_or_path gpt2 \
--dataset_name squad \
--do_train \
--do_eval \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--seed 42 \
--learning_rate 3e-03 \
--logging_strategy steps \
--logging_steps 10 \
--lr_scheduler_type linear \
--report_to tensorboard \
--gradient_accumulation_steps 1 \
--max_seq_length 384 \
--doc_stride 128 \
--save_safetensors False \
--max_steps 1000 \
--w_bits 32 6 \
--a_bits 32 6 \
--kv_bits 32 6 \
--lora_attn_dim 16 \
--lora_attn_alpha 8 \
--lora_dropout 0.0 \
--lora_num_per_layer 2 \
--output_dir ./result 
```

3. Run ***eval_qa.py*** to flexibly switch precision configurations during inference. This example code activates the 32 bit-width configuration for LoRA Modules applied to the final classification layer, and activates 8 bit-width configuration for LoRA modules applied to the 12 attention blocks in GPT-2:
```
python eval_qa.py --model_name_or_path result \
--dataset_name squad \
--do_eval \
--activate_lora_idx 1 1 1 1 1 1 1 1 1 1 1 1 0 \
--output_dir ./evaluation/
```
The **--activate_lora_idx** decides which LoRA module to be activated for the 13 layers (12 attention layers + 1 output linear layer). In this example, LoRA module with index 0 refers to the one with 32 bit-width configuration as we input **--w_bits 32 6** for training.

### Attempt 3: LoRA finetuning of GPT-2 under cyclic precision 
1. Go to the project directory for cyclic precision LoRA training:
```
cd cyclic_precision
```

2. Run ***run_qa.py***. This example code enables cyclic precision for all the modules to be quantized in Step 1:
```
run_qa.py --model_name_or_path gpt2 \
--dataset_name squad \
--do_train \
--do_eval \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--seed 42 \
--learning_rate 3e-03 \
--num_train_epochs 2 \
--logging_strategy steps \
--logging_steps 10 \
--lr_scheduler_type linear \
--report_to tensorboard \
--gradient_accumulation_steps 1 \
--max_seq_length 384 \
--doc_stride 128 \
--save_safetensors False \
--max_steps 1000 \
--num_cyclic_period 32 \
--num_bit_min 16 \
--num_bit_max 32 \
--lora_attn_dim 16 \
--lora_attn_alpha 8 \
--lora_dropout 0.0 \
--output_dir ./results
```
With **--num_cyclic_period 32**, **--num_bit_min 16** and **--num_bit_max 32**, the quantization bit-width cycles from 16 to 32, and then from 32 to 16 for 32 times before the end of training.


### Attempt 4: Adversarial attack under switchable precision
1. Go to the project directory for adversarial attack:
```
cd adversarial_attack
```

2. Run ***attack.py***. This example code generates adversarial samples by GPT-2 model with LoRA modules quantized by 16 bit-width configurations, and then attacks its 6 bit-width LoRA alternative modules:
```
cd adversarial_attack
```

The output of this command is the test accuracy achieved by LoRA modules under the 6 bit-with configuration after the attack. This step aims to evaluate the transferability of adversarial attacks across different bit-width configurations. 


## Results

- to be done


