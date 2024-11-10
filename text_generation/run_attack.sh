# gpt2

## wikitext

python3 attack.py --job_dir ../../llm_experiment/proj7_attack/gpt2/wikitext/attack_30s_val_0/ \
    --bitW 32 --abitW 32 \
    --finetuned True \
    --source_file ../../llm_experiment/gpt2/all/t_0/checkpoint/model_best.pt \
    --gumbel_samples 30 \
    --gpus 7 --lr 3e-2 \
    --method wikitext

## ptb

python3 attack.py --job_dir ../../llm_experiment/proj7_attack/gpt2/ptb/attack_30s_val_0/ \
    --bitW 32 --abitW 32 \
    --finetuned True \
    --source_file ../../llm_experiment/gpt2/all/t_0/checkpoint/model_best.pt \
    --gumbel_samples 30 \
    --gpus 7 --lr 3e-2 \
    --method ptb

## c4

python3 attack.py --job_dir ../../llm_experiment/proj7_attack/gpt2/c4/attack_30s_val_0/ \
    --bitW 32 --abitW 32 \
    --finetuned True \
    --source_file ../../llm_experiment/gpt2/all/t_0/checkpoint/model_best.pt \
    --gumbel_samples 30 \
    --gpus 7 --lr 3e-2 \
    --method c4


