'''
Usage:
python -m mix_eval.evaluate \
    --model_name {gpt_4_turbo_2024_04_09, llama_3_8b_instruct, ...} \
    --benchmark {mixeval, mixeval_hard} \
    --version VERSION \
    --batch_size BATCH_SIZE \ 
    [--max_gpu_memory MAX_GPU_MEMORY] \ 
    [--data_path DATA_PATH_FREEFORM] \ 
    [--output_dir OUTPUT_FOLDER] \ 
    [--verbose]
'''
import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import warnings
from tqdm import tqdm
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)
import json
from torch.utils.data import DataLoader
from loguru import logger
import concurrent.futures

import MixEval.mix_eval.api.registry
#from MixEval.mix_eval.models import AVAILABLE_MODELS
from MixEval.mix_eval.utils.dataset import get_eval_dataset
from MixEval.mix_eval.compute_metrics import compute_metrics_p
from MixEval.mix_eval.utils.common_utils import (
    set_seed, 
    cache_status, 
    read_status, 
    dict_equal,
    log_error
    )

from MixEval.mix_eval.prompts.evaluation_prompts import (
    FIVE_SHOT_PREFIX_FREEFORM,
    FIVE_SHOT_PREFIX_MULTIPLECHOICE
    )


from typing import List

from utils import (
    generate_together,
    generate_openai,
    generate_with_references,
    DEBUG,
)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--reference_models", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--provider", type=str, default="together")

    parser.add_argument(
        "--num_choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--force_temperature", type=float, help="Forcibly set a sampling temperature."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    
    parser.add_argument(
        "--benchmark", 
        type=str, 
        required=True, 
        choices=["mixeval", "mixeval_hard"], 
        help="Benchmark to evaluate."
        )
    parser.add_argument(
        "--version", 
        type=str, 
        required=True,
        help="The benchmark version to run. We update MixEval data points on a monthly basis."
        )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        required=True, 
        help="Batch size for evaluation."
        )
    parser.add_argument(
        "--max_gpu_memory", 
        type=str, 
        default=None, 
        help="The maximum memory per GPU for storing model weights. "
        "Set this properly will allocate more memory for activations, "
        "so you can use longer context lengths or larger batch sizes."
        )
    parser.add_argument(
        "--api_parallel_num", 
        type=int, 
        default=100, 
        help="Number of parallel threads for calling the model parser api if use model parsing." 
        "If you hit rate limit error frequently, try to reduce this number."
        )
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="mix_eval/data/", 
        help="Path to benchmark data. It should be the parent dir of the dir containing mixeval/ and mixeval_hard/."
        )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="mix_eval/data/model_responses", 
        help="Path to save model responses."
    )
    parser.add_argument(
        "--inference_only", 
        action="store_true", 
        help="If set this flag, it will generate model responses only without computing the scores."
        )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Print verbose information."
        )
    parser.add_argument(
        "--free_form_parser", 
        type=str, 
        default="model", 
        choices=["model", "rule"], 
        help="Parser for freeform responses, either model parser or rule-based parser.")
    parser.add_argument(
        "--multi_choice_parser", 
        type=str, 
        default="model", 
        choices=["model", "rule"], 
        help="Parser for multiple-choice responses, either model parser or rule-based parser."
        )
    parser.add_argument(
        "--multichoice_judge",
        type=str, 
        default="gpt-3.5-turbo-0125", 
        help="Judge model for multiple-choice score computation."
        )
    parser.add_argument(
        "--freeform_judge",
        type=str, 
        default="gpt-3.5-turbo-0125", 
        help="Judge model for freeform score computation."
        )
    parser.add_argument(
        "--extract_base_model_response", 
        action="store_true", 
        help="The unfinetuned models will produce endless output, "
        "which may influence the model parse score."
        )
    parser.add_argument(
        "--compute_score_from_judged_file", 
        action="store_true", 
        help="Whether to compute score directly from the judged file."
        "This will save budge for those models that has been judged before."
        "it also helps to do some analysis easily without running judgements again."
        )
    return parser.parse_args()

def get_answer(
    question,
    model: str,
    reference_models: List[str],
    num_choices: int,
    max_tokens: int,
    rounds: int,
    provider: str,
):

    if args.force_temperature is not None:
        temperature = args.force_temperature
    else:
        temperature = 0.0

    if provider == "together":
        generate_fn = generate_together
    elif provider == "openai":
        generate_fn = generate_openai
    else:
        assert False
    messages = []

    qs = question

    messages.append({"role": "user", "content": qs})

    references = []

    if len(reference_models) > 0:

        prev_references = []

        for i_round in range(rounds):

            if DEBUG:
                logger.info(
                    f"Round {i_round+1}/{rounds} to collecting reference responses."
                )

            references = []

            for reference_model in reference_models:

                reference = generate_with_references(
                    model=reference_model,
                    messages=messages,
                    references=prev_references,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    generate_fn=generate_fn,
                )

                if reference is not None:

                    references.append(reference)

            if i_round < rounds - 1:

                prev_references = references

                references = []

    output = generate_with_references(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        generate_fn=generate_fn,
        references=references
    )

    return output.strip()

def get_responses(batch, response_file):

    if args.reference_models is None:
        reference_models = []
    else:
        reference_models = args.reference_models.split(",")

    formated_prompts = [d['raw_inputs']['formated_input'] for d in batch]
        
    # add few-shot prompts
    if args.split == 'close_multichoice' or args.split == 'close_multichoice_hard':
        formated_prompts = [
                            FIVE_SHOT_PREFIX_MULTIPLECHOICE + prompt + '\n' 
                            for prompt in formated_prompts
                            ]
    elif args.split == 'close_freeform' or args.split == 'close_freeform_hard':
        formated_prompts = [
                            FIVE_SHOT_PREFIX_FREEFORM + prompt + '\n' 
                            for prompt in formated_prompts]

    for _fp, _b in zip(formated_prompts, batch):
        _b['raw_inputs']['formated_input'] = _fp

    responses = [None] * len(formated_prompts)
    futures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.api_parallel_num) as executor:
        
        for i, question in enumerate(formated_prompts):
            future = executor.submit(
                    get_answer,
                    question,
                    args.model_name,
                    reference_models,
                    args.num_choices,
                    args.max_tokens,
                    args.rounds,
                    args.provider,
                )
            futures.append((i, future))

        for i, future in futures:
            result = future.result()
            responses[i] = result

    with open(response_file, "a") as f:
            for raw_dict, response in zip(batch, responses):
                raw_dict = raw_dict['raw_inputs']
                raw_dict['response'] = response
                f.write(json.dumps(raw_dict) + "\n")

def _eval(args):
    print(f"\n\nStart to evaluate {args.model_name}'s {args.split} split. \n\n")
    time_elapsed = 0
    start_time = time.time()
    
    response_file = os.path.join(
        args.output_dir, 
        args.model_name, 
        args.benchmark,
        args.version,
        f"{args.model_name}_{args.split}.jsonl"
        )
    os.makedirs(
        os.path.dirname(response_file), 
        exist_ok=True
        )
    
    # if the response file exists, check if it can resume from last run
    resume = False
    if os.path.exists(response_file):
        status = read_status(args)
        if not dict_equal(status['args'], args.__dict__):
            raise ValueError(f"The model response file {response_file} already exists. The cached arguments are "
                            "different from those in the current run. Please check.")
        if status['status']['status'] == 'complete':
            print(f"The evaluation for {args.model_name}'s {args.split} "
                    "split is already complete. Skipping.")
            return
        with open(response_file) as f:
            lines = f.readlines()
            if len(lines) == (status['status']['batch_id'] + 1) * args.batch_size:
                resume = True
                time_elapsed += time.time() - start_time + status['status']['time_elapsed']
                start_time = time.time()
                print(f"Resuming from last run: \n{status}")
            else:
                raise ValueError(f"The response file [{response_file}] has different "
                                "lines as recorded in cached metadadta. Please check the response file. "
                                "You might consider delete the response and metadata file to start from scratch.")
    
    #model = mix_eval.api.registry.get_model(args.model_name)(args)
    eval_dataset = get_eval_dataset(args)
    dataloader = DataLoader(
        eval_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=8, 
        collate_fn=lambda x: x
        )
    
    for b_id, batch in enumerate(tqdm(dataloader, desc="Evaluating batches", unit="batch")):
        if resume:
            if status['status']['batch_id'] >= b_id:
                continue
            else:
                resume = False

        if args.verbose:
            _start_time = time.time()

        get_responses(batch, response_file)


        if args.verbose:
            _finish_time = time.time()
            print(f"Batch {b_id} finished in {_finish_time - _start_time} seconds.")
        
        time_elapsed += time.time() - start_time
        start_time = time.time()
        
        status = {
            'batch_id': b_id,
            'time_elapsed': time_elapsed,
            'status': 'in progress'
        }
        cache_status(args, status)

    status = {
        'batch_id': b_id,
        'time_elapsed': time_elapsed,
        'status': 'complete'
    }
    cache_status(args, status)
    print(f"Finished evaluating {args.model_name}'s {args.split} split. "
          f"Used {round(time_elapsed / 60, 2)} minutes.")


def eval(args):
    if args.benchmark == "mixeval":
        args.split = "close_freeform"
        _eval(args)
        args.split = "close_multichoice"
        _eval(args)
    elif args.benchmark == "mixeval_hard":
        args.split = "close_freeform_hard"
        _eval(args)
        args.split = "close_multichoice_hard"
        _eval(args)
    else:
        raise ValueError(f"Benchmark {args.benchmark} not supported.")

if __name__ == '__main__':
    set_seed()
    args = parse_args()
    try:
        eval(args)
        if not args.inference_only:
            compute_metrics_p(args)
    except Exception as e:
        msg = (f"Error: {e}; Model: {args.model_name}; "
        f"Split: {args.split}; "
        f"Check the logfile: {args.output_dir}/{args.model_name}/"
        f"{args.benchmark}/{args.version}/{args.model_name}.log")
        log_error(msg, f"{args.output_dir}/error.log")
        raise e
    
