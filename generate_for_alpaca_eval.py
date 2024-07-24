import json
import datasets
from fire import Fire
from functools import partial
from typing import List
from loguru import logger
import time

from utils import (
    generate_together,
    generate_openai,
    generate_with_references,
    generate_reference_models,
    generate_layer_output,
    generate_branch_output,
    DEBUG,
)


def process_fn(
    item,
    model,
    reference_models=[],
    temperature=0.7,
    max_tokens=2048,
    rounds=1,
    branches=0,
    aggregate_temp=0.0,
    tag="",
):

    messages = [{"role": "user", "content": item["instruction"]}]

    references = item.get("references", [])

    if branches > 0:
        output = generate_branch_output(
            model=model,
            reference_models=reference_models,
            references=references,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            rounds=rounds,
            branches=branches,
            aggregate_temp=aggregate_temp,
        )
    else:
        output = generate_layer_output(
            model=model,
            reference_models=reference_models,
            references=references,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            rounds=rounds,
        )

    return {"output": output, "generator": model + "-together-" + tag}


def main(
    model: str,
    output_path: str,
    reference_paths: str = None,
    reference_models: str = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    rounds: int = 1,
    num_proc: int = 16,
    branches: int = 0,
    aggregate_temp: float = 0.0,
    tag: str = "",
):

    print(reference_models)

    if reference_paths is None:
        reference_paths = []
    else:
        reference_paths = reference_paths.split(",")

    if reference_models is None:
        reference_models = []
    else:
        reference_models = reference_models.split(",")

    eval_set = datasets.load_dataset(
        "tatsu-lab/alpaca_eval", "alpaca_eval_gpt4_baseline", trust_remote_code=True
    )["eval"]
    eval_set = eval_set.remove_columns(["output", "generator"])

    if len(reference_paths):

        logger.info(f"`reference_paths` provided: {reference_paths}")

        references = []
        for reference_path in reference_paths:
            with open(reference_path) as f:
                reference_responses = json.load(f)
                logger.info(
                    f"Reading reference outputs: {reference_path} ({len(reference_responses)})"
                )
                for i_reference_response, reference_response in enumerate(
                    reference_responses
                ):
                    if len(references) <= i_reference_response:
                        references.append([reference_response["output"]])
                    else:
                        references[i_reference_response].append(
                            reference_response["output"]
                        )

        eval_set = eval_set.add_column(f"references", references)
        print(eval_set)

    if len(reference_models):
        logger.info(
            f"`reference_models` provided: {reference_models}. Will generate reference responses on-the-fly."
        )

    logger.info(f"Start.")

    eval_set = eval_set.map(
        partial(
            process_fn,
            model=model,
            reference_models=reference_models,
            temperature=temperature,
            max_tokens=max_tokens,
            rounds=rounds,
            branches=branches,
            aggregate_temp=aggregate_temp,
            tag=tag,
        ),
        batched=False,
        num_proc=num_proc,
    )

    logger.info(f"Saving outputs to {output_path}.")

    try:
        eval_set = eval_set.remove_columns(f"references")
    except Exception as e:
        pass

    with open(output_path, "w") as f:

        json.dump(list(eval_set), f, indent=2)


if __name__ == "__main__":

    Fire(main)
