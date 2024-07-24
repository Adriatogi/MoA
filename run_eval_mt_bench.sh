export DEBUG=1

# python3 generate_for_mt_bench.py --model "mistralai/Mistral-7B-Instruct-v0.3" \
#     --reference-models "snorkelai/Snorkel-Mistral-PairRM-DPO,mistralai/Mistral-7B-Instruct-v0.3,NousResearch/Nous-Hermes-2-Mistral-7B-DPO,togethercomputer/StripedHyena-Nous-7B,Qwen/Qwen1.5-7B-Chat,togethercomputer/Llama-2-7B-32K-Instruct" \
#     --answer-file outputs/mt_bench/Mistral-7B-Instruct-v0.3-MoA-round1.jsonl \
#     --parallel 16 --rounds 1

# python3 generate_for_mt_bench.py --model "Qwen/Qwen1.5-110B-Chat" \
#     --reference-models "snorkelai/Snorkel-Mistral-PairRM-DPO,mistralai/Mistral-7B-Instruct-v0.3,NousResearch/Nous-Hermes-2-Mistral-7B-DPO,togethercomputer/StripedHyena-Nous-7B,Qwen/Qwen1.5-7B-Chat,togethercomputer/Llama-2-7B-32K-Instruct" \
#     --answer-file outputs/mt_bench/Qwen1.5-110B-Chat-MoA-7b-round1.jsonl \
#     --parallel 16 --rounds 1

# python3 generate_for_mt_bench.py --model "snorkelai/Snorkel-Mistral-PairRM-DPO" \
#     --answer-file outputs/mt_bench/Snorkel-Mistral-PairRM-DPO.jsonl \
#     --parallel 16 --rounds 3

# python3 generate_for_mt_bench.py --model "mistralai/Mistral-7B-Instruct-v0.3" \
#     --answer-file outputs/mt_bench/Mistral-7B-Instruct-v0.3.jsonl \
#     --parallel 16 --rounds 3

# python3 generate_for_mt_bench.py --model "NousResearch/Nous-Hermes-2-Mistral-7B-DPO" \
#     --answer-file outputs/mt_bench/Nous-Hermes-2-Mistral-7B-DPO.jsonl \
#     --parallel 16 --rounds 3

# python3 generate_for_mt_bench.py --model "togethercomputer/StripedHyena-Nous-7B" \
#     --answer-file outputs/mt_bench/StripedHyena-Nous-7B.jsonl \
#     --parallel 16 --rounds 3

# python3 generate_for_mt_bench.py --model "Qwen/Qwen1.5-7B-Chat" \
#     --answer-file outputs/mt_bench/Qwen1.5-7B-Chat.jsonl \
#     --parallel 16 --rounds 3

# python3 generate_for_mt_bench.py --model "Qwen/Qwen1.5-110B-Chat" \
#     --reference-models "microsoft/WizardLM-2-8x22B,Qwen/Qwen1.5-110B-Chat,Qwen/Qwen1.5-72B-Chat,meta-llama/Llama-3-70b-chat-hf,mistralai/Mixtral-8x22B-Instruct-v0.1,databricks/dbrx-instruct" \
#     --answer-file outputs/mt_bench/Qwen1.5-110B-Chat-MoA-round3.jsonl \
#     --parallel 16 --rounds 3

# python3 generate_for_mt_bench.py --model "Qwen/Qwen1.5-72B-Chat" \
#     --reference-models "microsoft/WizardLM-2-8x22B,Qwen/Qwen1.5-110B-Chat,Qwen/Qwen1.5-72B-Chat,meta-llama/Llama-3-70b-chat-hf,mistralai/Mixtral-8x22B-Instruct-v0.1,databricks/dbrx-instruct" \
#     --answer-file outputs/mt_bench/Qwen1.5-72B-Chat-MoA-lite-round1.jsonl \
#     --parallel 16 --rounds 1
# python3 eval_mt_bench.py --model-list Qwen1.5-72B-Chat-MoA-lite-round1 --parallel 32


#python3 eval_mt_bench.py --model-list Mistral-7B-Instruct-v0.3-MoA-round1 Qwen1.5-110B-Chat-MoA-7b-round1 --parallel 32

# python3 eval_mt_bench.py --model-list Mistral-7B-Instruct-v0.3 --parallel 32

# python3 eval_mt_bench.py --model-list Nous-Hermes-2-Mistral-7B-DPO --parallel 32

# python3 eval_mt_bench.py --model-list StripedHyena-Nous-7B --parallel 32

# python3 eval_mt_bench.py --model-list Qwen1.5-7B-Chat --parallel 32


# python3 eval_mt_bench.py --model-list Qwen1.5-110B-Chat-MoA-round3 --parallel 32

python3 generate_for_mt_bench.py --model "Qwen/Qwen1.5-110B-Chat" \
    --reference-models "microsoft/WizardLM-2-8x22B,Qwen/Qwen1.5-110B-Chat,Qwen/Qwen1.5-72B-Chat,meta-llama/Llama-3-70b-chat-hf,mistralai/Mixtral-8x22B-Instruct-v0.1,databricks/dbrx-instruct" \
    --answer-file outputs/mt_bench/Qwen1.5-110B-MoA-stacked-round1-choices5.jsonl \
    --parallel 16 --rounds 1 --branches 2 --aggregate_temp 0.0
#python3 eval_mt_bench.py --model-list Qwen1.5-110B-MoA-stacked-round1-choices5 --parallel 32
#python3 eval_mt_bench.py --model-list Qwen1.5-7B-Chat --parallel 32
# python3 show_mt_bench_result.py