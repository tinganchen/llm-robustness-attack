# gpt2

## lambada

python3 attack.py --job_dir ../../llm_experiment/proj9_attack_tasks/gpt2/lambada/attack_30s_0/ \
    --bitW 32 --abitW 32 \
    --max_seq_len 256 \
    --finetuned True \
    --source_file ../../llm_experiment/gpt2/lambada/t_0/checkpoint/model_best.pt \
    --gumbel_samples 30 \
    --gpus 7 --lr 3e-2 \
    --method lambada

