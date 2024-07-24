import os
import json
import time
import requests
import openai
import copy

from loguru import logger


DEBUG = int(os.environ.get("DEBUG", "0"))


def generate_together(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
    streaming=False,
    completion_tokens=False
):

    output = None

    for sleep_time in [1, 2, 4, 8, 16, 32]:

        try:

            endpoint = "https://api.together.xyz/v1/chat/completions"

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}...`) to `{model}`."
                )

            res = requests.post(
                endpoint,
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": (temperature if temperature > 1e-4 else 0),
                    "messages": messages,
                },
                headers={
                    "Authorization": f"Bearer {os.environ.get('TOGETHER_API_KEY')}",
                },
            )
            if "error" in res.json():
                logger.error(res.json())
                if res.json()["error"]["type"] == "invalid_request_error":
                    logger.info("Input + output is longer than max_position_id.")
                    return None

            if completion_tokens:
                output = {"content": res.json()["choices"][0]["message"]["content"].strip(), "completion_tokens": res.json()["usage"]["completion_tokens"]}
            else:
                output = res.json()["choices"][0]["message"]["content"].strip()

            break

        except Exception as e:
            logger.error(e)
            if DEBUG:
                logger.debug(f"Msgs: `{messages}`")

            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    if output is None:
        return output

    if DEBUG:
        if completion_tokens:
            logger.debug("Output: `"+ output["content"][:20] + "...`.")
        else:
            logger.debug(f"Output: `{output[:20]}...`.")

    return output.strip()


def generate_together_stream(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):
    endpoint = "https://api.together.xyz/v1"
    client = openai.OpenAI(
        api_key=os.environ.get("TOGETHER_API_KEY"), base_url=endpoint
    )
    endpoint = "https://api.together.xyz/v1/chat/completions"
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature if temperature > 1e-4 else 0,
        max_tokens=max_tokens,
        stream=True,  # this time, we set stream=True
    )

    return response


def generate_openai(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):

    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = completion.choices[0].message.content
            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    output = output.strip()

    return output


def inject_references_to_messages(
    messages,
    references,
):

    messages = copy.deepcopy(messages)

    system = f"""You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

    for i, reference in enumerate(references):

        system += f"\n{i+1}. {reference}"

    if messages[0]["role"] == "system":

        messages[0]["content"] += "\n\n" + system

    else:

        messages = [{"role": "system", "content": system}] + messages

    return messages


def generate_with_references(
    model,
    messages,
    references=[],
    max_tokens=2048,
    temperature=0.7,
    generate_fn=generate_together,
    completion_tokens=False
):

    if len(references) > 0:

        messages = inject_references_to_messages(messages, references)

    return generate_fn(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        completion_tokens=completion_tokens
    )

def generate_reference_models(messages, reference_models,
                              temperature, max_tokens,
                              rounds, generate_fn=generate_together):
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

    return references

def generate_layer_output(model,
                          reference_models,
                          messages,
                          max_tokens,
                          temperature,
                          rounds,
                          generate_fn=generate_together):
    references = []

    # generate refrences
    if len(reference_models) > 0:

        references = generate_reference_models(messages=messages,
                                               reference_models=reference_models,
                                               temperature=temperature,
                                               max_tokens=max_tokens,
                                               generate_fn=generate_fn,
                                               rounds=rounds)

    # aggregate on top of refrences
    output = generate_with_references(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        generate_fn=generate_fn,
        references=references,
    )

    return output

def generate_branch_output(model,
                           reference_models,
                           messages,
                           max_tokens,
                           temperature,
                           rounds,
                           branches,
                           aggregate_temp=0.0,
                           generate_fn=generate_together):
    branch_responses = []
    for k in range(branches):
        print("branch ", k)
        output = generate_layer_output(model=model, 
                                    reference_models=reference_models,
                                    messages=messages,
                                    max_tokens=max_tokens, 
                                    temperature=temperature,
                                    generate_fn=generate_fn, 
                                    rounds=rounds)
        if output is not None:
            branch_responses.append(output)

    output = generate_with_references(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=aggregate_temp,
        generate_fn=generate_fn,
        references=branch_responses,
        )
    return output