# LLM Project - Analysis on Robustness of Efficient LLM Models under Attack (@NVIDIA Research Team, 2024)
[Background] 

* Language models suffer from large memory and computation costs
* Recent research have been proposed to compress the models for memory efficiency and acceleration, such as quantization which reduces the precisions of weights and activations

[Challenge]

* Is the quantization process robust? [No]
* Under the experiments, the models compressed to lower bits are more sensitive to the attack on the text data
* The performance degrades drastically at low bits


## Requirements

* python3
* pytorch==1.7.1
* cudatoolkit==11.0.221 
* numpy==1.19.2
* tensorboardx==1.4

## Experiments

Task                | LLM Models               | Evaluation Metric   | Datasets  
---                  |---                  |---                                    |---                    
Text Generation |GPT-2 (124M) & OPT (1.3B)           | Perplexity                                    | WikiText-2 & PTB & C4           
Word Prediction |GPT-2 (124M)           | Accuracy (%)                                   | Lambada

## Implementation

### Task - Word Prediction (eg. GPT-2 on Lambada)

#### Inference of low-bit models

```shell
cd word_prediction/
bash run.sh
```

#### Attack 

```shell
bash run_attack.sh
```

### Evaluation accuracy of attacked models

```shell
bash run_attack_eval.sh
```

## Results

* External Link - [Drive](https://docs.google.com/spreadsheets/d/1oGmjS9yNzh38bQXJLVJVCkQc5RHDuWNe5X6nsNzpZKQ/edit?usp=sharing) 

## References

* Models & Datasets - [HuggingFace]([https://arxiv.org/abs/1903.09291](https://huggingface.co/))
* Adversarial Attack - [Paper](https://aclanthology.org/2021.emnlp-main.464/) 
