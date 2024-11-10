# gpt2

## lambada
python main.py \
    --model gpt2 --target_model GPT2LMHeadModel \
    --method lambada --max_seq_len 1024 \
    --job_dir ../../../llm_experiment/gpt2/lambada/t_0/ \
    --source_file ../../../llm_experiment/gpt2/all/t_0/ \
    --finetuned False \
    --gpus 7 --lr 1e-4
    