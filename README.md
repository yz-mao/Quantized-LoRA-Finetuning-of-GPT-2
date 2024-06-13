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
- Create a new conda environment:
```
conda env create -f env.yml
conda activate GPT2
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
--w_bits 8 \
--a_bits 8 \
--kv_bits 8 \
--output_dir ./results
```

The layers to be quantized when running this code include:
- The query/key/value projection in each attention block (quantized by **w_bits**)
- The KV cache (quantized by **kv_bits**)
- The output projection in each attention block (quantized by **w_bits**)
- The two linear layers in each MLP module (quantized by **w_bits**)
- The final linear layer (quantized by **w_bits**)
- The inputs for these layers above (quantized by **a_bits**)


### Attempt 2: LoRA finetuning of GPT-2 under switchable precision 
1. Go to the project directory for switchable precision LoRA training:
```
cd switchable_precision
```

2. Run ***run_qa.py***. This example code fine-tunes GPT-2 on the SQuAD 1.0 dataset with 2 LoRA modules (2 pairs of LoRA adapters with rank 16) per layer, one of which is quantized by 16 bits per value and the other is quantized by 8 bits per value: 
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
--w_bits 16 8 \
--a_bits 16 8 \
--kv_bits 16 8 \
--lora_attn_dim 16 \
--lora_attn_alpha 8 \
--lora_dropout 0.0 \
--lora_num_per_layer 2 \
--output_dir ./results_16_8 
```

3. Run ***eval_qa.py*** to flexibly switch precision configurations during inference. This example code activates the 16-bit width configuration for LoRA Modules applied to the final classification layer and the 8-bit width configuration for LoRA modules applied to each of the 12 attention blocks in GPT-2:
```
python eval_qa.py --model_name_or_path results_16_8 \
--dataset_name squad \
--do_eval \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--seed 42 \
--report_to tensorboard \
--activate_lora_idx 1 1 1 1 1 1 1 1 1 1 1 1 0 \
--output_dir ./evaluation/
```
The **--activate_lora_idx** parameter determines which LoRA module to activate for the 13 layers (12 attention layers + 1 output linear layer). In this example, the LoRA module with index 0 refers to the one with a 16-bit width configuration, as indicated by the **--w_bits 16 8** input during training.


### Attempt 3: LoRA finetuning of GPT-2 under cyclic precision 
1. Go to the project directory for cyclic precision LoRA training:
```
cd cyclic_precision
```

2. Run ***run_qa.py***. This example code enables cyclic precision for all the modules to be quantized in Attempt 1:
```
python run_qa.py --model_name_or_path gpt2 \
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

2. Run ***attack.py***. This example code generates adversarial samples using a GPT-2 model with LoRA modules quantized to 16-bit width configurations, and then attacks its alternative LoRA modules quantized to 8-bit width:
```
python attack.py --model_name_or_path results_16_8 \
--dataset_name squad \
--do_train
--do_eval
--per_device_train_batch_size 16
--per_device_eval_batch_size 16
--seed 42
--learning_rate 3e-03
--logging_strategy steps
--logging_steps 10
--lr_scheduler_type linear
--report_to tensorboard
--gradient_accumulation_steps 1
--max_seq_length 384
--doc_stride 128
--max_steps 10
--attack_lora_idx 0
--eval_lora_idx 1
--adversarial_generation_step 10
--adversarial_sample_num 100
--output_dir ./attack_results
```

The output of this command is the test accuracy achieved by LoRA modules under the 8 bit-with configuration after the attack. This step aims to evaluate the transferability of adversarial attacks across different bit-width configurations. 


## Results

- Table 1: Training results with static quantization
  
| **Model** | **Bit-Width Configurations** | **EM** | **F1 Score** | **Number of Trainable Parameters** |
|-----------|-------------------------------|--------|--------------|------------------------------------|
| **GPT-2** | Full Precision (fp)           | 63.302 | 74.406       | 124,441,346                        |
| **GPT-2** | 16 bit                        | 61.410 | 72.756       | 124,441,346                        |
| **GPT-2** | 8 bit                         | 60.341 | <u>71.915</u>| 124,441,346                        |
| **GPT-2** | 6 bit                         | 56.216 | 68.870       | 124,441,346                        |
| **GPT-2** | 5 bit                         | 42.148 | 56.092       | 124,441,346                        |

- Table 2: Training results with multiple bit-width configurations per layer
  
| **Model**                          | **Inference Bit-Width Configurations**              | **EM** | **F1 Score** |
|------------------------------------|------------------------------------------------------|--------|--------------|
| **GPT-2 + 8 / 6 bit LoRA**         | [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]              | 59.044* | 71.094*      |
| **GPT-2 + 8 / 6 bit LoRA**         | [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]              | 53.671* | **66.569***  |
| **GPT-2 + 8 / 6 bit LoRA**         | [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]              | 59.045  | <u>71.094</u>|
| **GPT-2 + 8 / 6 bit LoRA**         | [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]              | 54.683  | **67.389**   |
| **GPT-2 + 16 / 6 bit LoRA**        | [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16] | 60.577  | 72.222       |
| **GPT-2 + 16 / 6 bit LoRA**        | [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]              | 56.093  | **68.517**   |
| **GPT-2 + fp / 6 bit LoRA**        | [fp, fp, fp, fp, fp, fp, fp, fp, fp, fp, fp, fp, fp] | 62.517  | 73.961       |
| **GPT-2 + fp / 6 bit LoRA**        | [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]              | 57.398  | **69.910**   |

- Table 3: Training results with switchable precision configurations
  
| **Model**                          | **Inference Bit-Width Configurations**             | **EM** | **F1 Score** |
|------------------------------------|----------------------------------------------------|--------|--------------|
| **GPT-2 + 8 / 6 bit LoRA**         | [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, **6**]        | 43.094 | 59.182       |
| **GPT-2 + 8 / 6 bit LoRA**         | [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, **6**, 8]        | 57.606 | 69.984       |
| **GPT-2 + 8 / 6 bit LoRA**         | [**6**, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]        | 58.770 | **70.822**   |
| **GPT-2 + 8 / 6 bit LoRA**         | [**6, 6, 6, 6, 6, 6**, 8, 8, 8, 8, 8, 8, 8]        | 54.087 | 66.283       |
| **GPT-2 + 8 / 6 bit LoRA**         | [8, 8, 8, 8, 8, 8, **6, 6, 6, 6, 6, 6**, 8]        | 50.341 | 64.155       |
| **GPT-2 + 8 / 6 bit LoRA**         | [**6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6**, 8]        | 51.570 | **65.346**   |
| **GPT-2 + 16 / 6 bit LoRA**        | [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, **6**] | 17.966 | 33.816       |
| **GPT-2 + 16 / 6 bit LoRA**        | [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, **6**, 16] | 59.991 | 72.007       |
| **GPT-2 + 16 / 6 bit LoRA**        | [**6**, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16] | 60.407 | **72.031**   |
| **GPT-2 + 16 / 6 bit LoRA**        | [**6, 6, 6, 6, 6, 6**, 16, 16, 16, 16, 16, 16, 16] | 55.658 | 67.902       |
| **GPT-2 + 16 / 6 bit LoRA**        | [16, 16, 16, 16, 16, 16, **6, 6, 6, 6, 6, 6**, 16] | 53.377 | 66.091       |
| **GPT-2 + 16 / 6 bit LoRA**        | [**6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6**, 16]       | 53.113 | **66.494**   |

- Table 4: Training results with cyclic precision quantization
  
| **Model**                              | **Bit-Width Configurations** | **EM**  | **F1 Score**  | **# of Trainable Parameters** |
|----------------------------------------|------------------------------|---------|---------------|-------------------------------|
| **GPT-2 + Single LoRA**                | Full Precision (fp)          | 62.517  | 73.961        | 2,666,528                     |
| **GPT-2 + Single LoRA**                | 16 bit                       | 61.145  | 72.748        | 2,666,528                     |
| **GPT-2 + Single LoRA**                | 12 bit                       | 61.220  | **72.704**    | 2,666,528                     |
| **GPT-2 + Single LoRA**                | 16 ~ 32 bit                  | 60.558  | 72.195        | 2,666,528                     |
| **GPT-2 + Single LoRA**                | 8 ~ 16 bit                   | 59.413  | **71.401**    | 2,666,528                     |
| **GPT-2 + Single LoRA**                | 6 ~ 16 bit                   | 58.524  | 70.183        | 2,666,528                     |

- Table 5: Training results under adversarial attacks

| **Model**                    | **Attack Precision** | **Inference Precision** | **EM**  | **F1 Score**  |
|------------------------------|----------------------|-------------------------|---------|---------------|
| **GPT-2 + fp / 6 bit LoRA**  | **No Attack**        | fp                      | 70.000  | _75.178_      |
| **GPT-2 + fp / 6 bit LoRA**  | **No Attack**        | 6 bit                   | **63.000** | 70.592        |
| **GPT-2 + fp / 6 bit LoRA**  | **fp**               | fp                      | 67.000  | _70.307_      |
| **GPT-2 + fp / 6 bit LoRA**  | **fp**               | 6 bit                   | **66.000** | 68.494        |
| **GPT-2 + fp / 6 bit LoRA**  | **6 bit**            | fp                      | 67.000  | _70.307_      |
| **GPT-2 + fp / 6 bit LoRA**  | **6 bit**            | 6 bit                   | **65.000** | 67.846        |
| **GPT-2 + 16 / 6 bit LoRA**  | **No Attack**        | 16 bit                  | 66.000  | _70.263_      |
| **GPT-2 + 16 / 6 bit LoRA**  | **No Attack**        | 6 bit                   | 59.000  | **63.564**    |
| **GPT-2 + 16 / 6 bit LoRA**  | **16 bit**           | 16 bit                  | 63.000  | _64.605_      |
| **GPT-2 + 16 / 6 bit LoRA**  | **16 bit**           | 6 bit                   | **66.000** | 68.492        |
| **GPT-2 + 16 / 6 bit LoRA**  | **6 bit**            | 16 bit                  | 63.000  | _64.605_      |
| **GPT-2 + 16 / 6 bit LoRA**  | **6 bit**            | 6 bit                   | **65.000** | 69.441        |

## Discussions
### Insights about Optimal Quantization Bit-Width Configurations

According to the results presented in Table 3, activating a low-precision LoRA module for the final linear layer results in a drastic drop in test accuracy. For example, by simply replacing the 8-bit-width LoRA module for adapting the final linear layer with the corresponding 6-bit-width LoRA module trained for the same layer, the F1 score decreases from 71.094 to 59.182. Meanwhile, activating a low-precision LoRA module (such as 6-bit-width) for the last attention block yields a lower F1 score (69.984) compared to activating a low-precision LoRA module for the first attention block (70.822), while maintaining high-precision LoRA modules (8-bit-width) for the remaining layers. A similar trend is also observed in the experiments on LoRA modules with 16-bit-width and 6-bit-width. It indicates that **deeper** layers in GPT-2 might require **higher** quantization precision to process fine-grained features. These findings are aligned with the observations of [AdaLoRA](https://arxiv.org/abs/2303.10512), where applying LoRA adapters to deeper layers yielded better fine-tuning performance.

Likewise, selecting low precision for the first half of attention blocks while maintaining high precision for the remaining half achieves better test accuracy than vice versa. Taking the GPT-2 adapted by two pairs of LoRA modules with 8-bit-width and 6-bit-width as an example again, selecting 6-bit-width LoRA modules for the first half of attention blocks achieves an F1 score of 66.283, which is higher than the 64.155 obtained with activating 6-bit-width LoRA modules for the second half of attention blocks. Moreover, activating 6-bit-width for all attention blocks achieves an F1 score of 65.346, which is just slightly lower than the 66.283 achieved by allocating higher precision 8-bit-width to the second half of attention blocks, but is more efficient considering the reduction in the total amount of bits used for quantizing the trainable parameters (For 8 / 6 bit LoRA, the reduction is from 18,678,016 to 16,023,808 bits, saving 14% of the bit-width budget; For 16 / 6 bit LoRA, the reduction is from 29,393,408 to 16,122,368 bits, saving 45.1% of the bit-width budget).

Therefore, when fine-tuning LLM under resource constraints, it might be reasonable to quantize the attention blocks more aggressively while keeping the final classification layer with high precision, or gradually increase the quantization precision level from shallow layers to deep layers. For future work, the aforementioned dynamic quantization method with increasing precision throughout the LLM might lead to a better efficiency-accuracy trade-off. Additionally, training objectives that encourage dynamic quantization while maintaining generalization may be helpful for determining the optimal quantization precision for each layer. It is very possible that the optimal quantization precision automatically selected for each layer also aligns with the observations that deeper layers need higher precision.


### Discrepancy Between CNN and LLM Performance with Cyclic Precision Training
According to the experimental results presented in Table 4, cyclic precision training shows no clear advantages in terms of test accuracy. For example, in the experiment where the bit-width ranges from 8 to 16, the quantization gradually increases from 8 to 16 bits during training. However, its final F1 score of 71.401 is lower than that of the counterpart with a static quantization bit-width of 12 bits (72.704), which achieves the same training efficiency.

This suggests that unlike in CNN model training, where larger quantization noise in the beginning stage aids in exploration, as reported in [CPT](https://arxiv.org/abs/2101.09868) method, such an effect is not observed in LLM training. Instead, the low precision in the beginning may lead to wide local minima that traps the training process, which is difficult to escape especially with higher precision and less noise in the latter stage. Further investigation into the differences in optimization landscapes between CNN models and LLMs is necessary for a deeper understanding.

### Transferability of Adversarial Attacks
Unlike the expected case where attacks target a specific precision and obviously degrade the model performance under the corresponding precision, attacks on either high-precision or low-precision models lead to noticeable performance degradation of the high-precision model. One potential reason is that high-precision LLM models are trained with fine-grained inputs and features, and thus are more sensitive to shifts in input distribution.

Although the aforementioned reason potentially explains why the performance of low-precision models is not negatively impacted by adversarial attacks as that of high-precision models, it is not sufficient to explain why the accuracy achieved by **"attacked"** low-precision models is even better compared to counterparts without attacks. This may be due to the fact that the adversarial samples generated by the chosen [adversarial attack method](https://arxiv.org/abs/2104.13733) are not effective enough. Notably, a filtering step is needed to select adversarial samples that successfully result in misclassification. However, no extra filtering is applied to the generated adversarial samples in this project. It is very likely that instead of being well-trained to mislead the target model, the resulting adversarial samples simply act as noisy samples for data augmentation. This particularly benefits the training of low-precision models, which are trained to process rough features.
