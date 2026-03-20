from datetime import datetime
import json
import random
import datasets
from dotenv import load_dotenv
import os
from fastapi import params
import torch
import gc
import re
from tqdm import tqdm
from vllm import vllm
import pickle
import argparse
import yaml
import os.path as osp
from src.utils import math_parser
from src.utils import load_dataset, get_question_answer
from src.utils.extractors import extract_answer, extract_trace
from src.utils.graders import grade_answer
from src.utils.prompts import REASONING_SYSTEM_PROMPTS
def generate_traces(
        model_name: str,
        dataset_name: str,
        verbose: bool = False,
        limit: int = None,
        **kwargs
    ):
    """
    Generate reasoning traces to study as artifacts.
    Arguments:
    - model_name: str, name or path of the model to use.
    - dataset_name: str, name of the dataset to use ('math', 'gpqa', 'connections').
    - verbose: bool, whether to print progress information.
    - kwargs: additional keyword arguments to pass to the vLLM
    """
    print("KWARGS")
    print(kwargs)
    user_env = os.getenv("USER")
    if user_env:
        env_path = f".{user_env}.env"
        print(f"Loading user-specific env: {env_path}")
    else:
        env_path = ".env"
        print("Loading default .env")
    hf_token = os.getenv("HF_TOKEN")
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
    if verbose:
        print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    system_prompt = REASONING_SYSTEM_PROMPTS.get(dataset_name, "")
    if verbose:
        print(f"Dataset {dataset_name} loaded with {len(dataset)} examples.")
        if system_prompt != "":
            print(f"Using system prompt for {dataset_name}")
    llama_user_prompt = {
        "math": "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem:\n\n",
    }
    chats = []
    questions = []
    answers = []
    length = len(dataset)
    if limit is not None:
        length = min(length, limit)
        dataset = dataset.select(range(length))
        if verbose:
            print(f"Limiting dataset to first {length} examples.")
    if "llama" in model_name.lower() and dataset_name in llama_user_prompt:
            print("Using LLaMA user prompt format.")
    for item in tqdm(dataset, total=length, desc="Creating chats"):
        question, answer = get_question_answer(dataset_name, item)
        if "llama" in model_name.lower() and dataset_name in llama_user_prompt:
            chats.append([
                {
                    "role": "user",
                    "content": llama_user_prompt[dataset_name] + question,
                }
            ])
        else:
            chats.append([
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": question,
                }
            ])
        questions.append(question)
        answers.append(answer)
    model_path = model_name
    torch.cuda.empty_cache()
    gc.collect()
    if verbose:
        print(f"Loading model from {model_path}")
    sampling_params = kwargs.pop("sampling_params", {"max_tokens": 8192, "seed": 42})
    params = vllm.SamplingParams(**sampling_params)
    print("PARAMS")
    print(params)
    model = vllm.LLM(model=model_path,
                     tokenizer=model_path,
                     hf_token=hf_token,
                     **kwargs
                    )
    if verbose:
        print("Model loaded")
        print(f"Running inference on {len(chats)} prompts")
    response = model.chat(chats, params)
    torch.cuda.empty_cache()
    gc.collect()
    if verbose:
        print("Inference completed")
    tokenizer = model.get_tokenizer()
    completions = []
    traces = []
    extracted_answers = []
    scores = []
    finished = []
    score_boosted = []
    for i, output in tqdm(enumerate(response), total=len(response), desc="Processing outputs"):
        completion = tokenizer.decode(output.prompt_token_ids + output.outputs[0].token_ids, skip_special_tokens=False)
        trace = extract_trace(completion, model_name)
        extracted_answer = extract_answer(completion, dataset_name, model_name)
        score = grade_answer(dataset_name, extracted_answer, answers[i])
        completions.append(completion)
        traces.append(trace)
        extracted_answers.append(extracted_answer)
        scores.append(score)
        if dataset_name == 'connections':
            score_boosted.append(score)
        else:
            score_boosted.append(1 if answers[i] in trace or score == 1 else 0)
        finished.append(output.finished)
    output = {
        "metadata": {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "collected_on": str(datetime.now()),
            "average_score": sum(scores) / len(scores),
            "average_boosted_score": sum(score_boosted) / len(scores),
            "system_prompt": system_prompt,
            "sampling_params": sampling_params,
            "kwargs": {k: str(v) for k, v in kwargs.items()},
        },
        "data": {
            "questions": questions,
            "completions": completions,
            "traces": traces,
            "extracted_answers": [json.dumps(a) for a in extracted_answers] if dataset_name == 'connections' else extracted_answers,
            "scores": scores,
            "scores_boosted": score_boosted,
            "ground_truth_answers": [json.dumps(a) for a in answers] if dataset_name == 'connections' else answers,
        }
    }
    if verbose:
        print("Finished!")
    return output
if __name__ == "__main__":
    random.seed(42)
    load_dotenv(f".{os.getenv('USER')}.env" if os.getenv("USER") else ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the model to load from Hugging Face.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Name of the dataset to load from disk.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/traces/",
        help="Directory to save generated traces.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print verbose logs.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/scripts/trace_config",
        help="Path to a config directory (YAML) with defaulty generation parameters.",
    )
    parser.add_argument(
        "-f", "--force_write",
        action="store_true",
        help="Whether to overwrite existing output files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of examples to process (for debugging).",
    )
    args = parser.parse_args()
    print("ARGS")
    print(args)
    assert args.model_name is not None, "Model name must be specified via --model_name."
    assert args.dataset_name is not None, "Dataset name must be specified via --dataset_name."
    config = {}
    assert args.config is not None, "Config path must be specified via --config."
    print(args.model_name.lower().replace("/", "--"), os.listdir(args.config))
    if args.model_name.lower().replace("/", "--") + ".yaml" in os.listdir(args.config):
        config_path = osp.join(args.config, args.model_name.lower().replace("/", "--") + ".yaml")
        with open(config_path, 'r') as f:
            if args.verbose:
                print(f"Loading config from {config_path}")
            config = yaml.safe_load(f)
    else:
        size = re.findall(r'(\d+\.?\d*)b', args.model_name)
        if args.verbose:
            print(f"Debug: Model name is {args.model_name}, size match is {size}")
        if size:
            size_str = float(size[-1].lower().replace("b", ""))
            if size_str > 50:
                if args.verbose:
                    print(f"Model size detected as {size_str}B, using large default config.")
                config_path = osp.join(args.config, "default_large.yaml")
            else:
                if args.verbose:
                    print(f"Model size detected as {size_str}B, using small default config.")
                config_path = osp.join(args.config, "default_small.yaml")
        else:
            if args.verbose:
                print(f"Model size could not be detected, using small default config.")
            config_path = osp.join(args.config, "default_small.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    call_kwargs = {}
    if isinstance(config.get("vllm_kwargs"), dict):
        call_kwargs.update(config["vllm_kwargs"])
    if "sampling_params" in config:
        call_kwargs["sampling_params"] = config["sampling_params"]
    if args.dataset_name in config.get("dataset_overrides", {}):
        if args.verbose:
            print(f"Applying dataset-specific overrides for {args.dataset_name}")
        dataset_overrides = config["dataset_overrides"][args.dataset_name]
        if "vllm_kwargs" in dataset_overrides:
            call_kwargs.update(dataset_overrides["vllm_kwargs"])
        if "sampling_params" in dataset_overrides:
            call_kwargs["sampling_params"] = dataset_overrides["sampling_params"]
    print("CALL KWARGS")
    print(call_kwargs)
    output_dir = osp.join(args.output_dir, args.dataset_name.replace("/", "_"))
    os.makedirs(output_dir, exist_ok=True)
    output_path = osp.join(
        output_dir,
        f"traces_{args.model_name.replace('/', '_')}.pkl"
    )
    archived_dir = osp.join(output_dir, "archived")
    os.makedirs(archived_dir, exist_ok=True)
    if os.path.exists(output_path):
        if not args.force_write:
            ts = int(datetime.now().timestamp())
            archived_path = osp.join(archived_dir, os.path.basename(output_path).replace(".pkl", f"_old_{ts}.pkl"))
            print(f"Warning: Output file {output_path} already exists, moving to {archived_path}.")
            os.rename(output_path, archived_path)
        else:
            print(f"Warning: Overwriting existing output file {output_path}.")
    output = generate_traces(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        verbose=args.verbose,
        limit=args.limit,
        **call_kwargs
    )
    with open(output_path, "wb") as f:
        pickle.dump(output, f)
    print(f"Generated traces saved to {output_path}")
