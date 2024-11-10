# facebook/opt-125m
'''
## wikitext-2
OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=8 main.py \
    --model facebook/opt-125m --target_model OPTClass \
    --method wikitext-2 --max_seq_len 2048 \
    --job_dir ../../llm_experiment/pretrain_0/opt-125m/wikitext-2/



## ptb
OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=8 main.py \
    --model facebook/opt-125m --target_model OPTClass \
    --method ptb --max_seq_len 2048 \
    --job_dir ../../llm_experiment/pretrain_0/opt-125m/ptb/



## c4
OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=8 main.py \
    --model facebook/opt-125m --target_model OPTClass \
    --method c4 --max_seq_len 2048 \
    --job_dir ../../llm_experiment/pretrain_0/opt-125m/c4/



## lambada
OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=8 main.py \
    --model facebook/opt-125m --target_model OPTClass \
    --method lambada --max_seq_len 2048 \
    --job_dir ../../llm_experiment/pretrain_0/opt-125m/lambada/
'''
# gpt2

## wikitext-2
OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=8 main.py \
    --model gpt2 --target_model GPT2LMHeadModel \
    --method wikitext-2 --max_seq_len 1024 \
    --job_dir ../../llm_experiment/pretrain_0/gpt2/wikitext-2/

## ptb
OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=8 main.py \
    --model gpt2 --target_model GPT2LMHeadModel \
    --method ptb --max_seq_len 1024 \
    --job_dir ../../llm_experiment/pretrain_0/gpt2/ptb/

## c4
OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=8 main.py \
    --model gpt2 --target_model GPT2LMHeadModel \
    --method c4 --max_seq_len 1024 \
    --job_dir ../../llm_experiment/pretrain_0/gpt2/c4/

## lambada
OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=8 main.py \
    --model gpt2 --target_model GPT2LMHeadModel \
    --method lambada --max_seq_len 1024 \
    --job_dir ../../llm_experiment/pretrain_0/gpt2/lambada/
