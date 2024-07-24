export DEBUG=1

# python3 generate_for_arena.py --model "Qwen/Qwen1.5-72-Chat" \
#      --reference-models "microsoft/WizardLM-2-8x22B,Qwen/Qwen1.5-110B-Chat,Qwen/Qwen1.5-72B-Chat,meta-llama/Llama-3-70b-chat-hf,mistralai/Mixtral-8x22B-Instruct-v0.1,databricks/dbrx-instruct" \
#      --answer-file outputs/arena-hard/arena-hard-together-MoA-round1.jsonl \
#      --parallel 16 --rounds 1

python3 generate_for_arena.py --model "Qwen/Qwen1.5-110B-Chat" \
    --reference-models "microsoft/WizardLM-2-8x22B,Qwen/Qwen1.5-110B-Chat,Qwen/Qwen1.5-72B-Chat,meta-llama/Llama-3-70b-chat-hf,mistralai/Mixtral-8x22B-Instruct-v0.1,databricks/dbrx-instruct" \
    --answer-file outputs/arena-hard/Qwen1.5-110B-Chat-MoA-round3.jsonl \
    --parallel 16 --rounds 3

# Make sure to transfer over generated answers to arena_hard_auto/data/arena-hard-v0.1/model_answer/
# As well, change judge configs
# cd arena_hard_auto/
# python3 gen_judgment.py 
# python3 show_result.py