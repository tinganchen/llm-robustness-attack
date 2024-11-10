# gpt2

## lambada

python3 attack_eval.py --job_dir ../../llm_experiment/proj9_attack_tasks/gpt2/lambada/attack_30s_val_0/ \
    --bitW 32 --abitW 32 \
    --max_seq_len 1024 \
    --finetuned True \
    --source_file ../../../llm_experiment/gpt2/lambada/t_0/ \
    --method lambada \
    --attack_ratio 0. \
    --gpus 7 --lr 3e-2 




