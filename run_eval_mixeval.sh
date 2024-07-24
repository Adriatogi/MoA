export MODEL_PARSER_API=$OPENAI_API_KEY
export DEBUG=0

python3 generate_for_mixeval.py --model_name "Qwen/Qwen1.5-110B-Chat" \
    --reference_models "microsoft/WizardLM-2-8x22B,Qwen/Qwen1.5-110B-Chat,Qwen/Qwen1.5-72B-Chat,meta-llama/Llama-3-70b-chat-hf,mistralai/Mixtral-8x22B-Instruct-v0.1,databricks/dbrx-instruct" \
    --api_parallel_num 12 --rounds 3 --benchmark "mixeval_hard" --version 2024-06-01 --batch_size 24 \
    --data_path MixEval/mix_eval/data --output_dir outputs/mixeval --inference_only --force_temperature 0.7

# python3 -m mix_eval.compute_metrics \
#     --benchmark mixeval \
#     --version 2024-06-01 \
#     --model_response_dir outputs/mixeval/ \
#     --api_parallel_num 32 \
#     --models_to_eval Qwen1.5-110B-Chat \
    

