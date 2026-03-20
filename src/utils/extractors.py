
import re
import sys
import json
import pandas as pd
import pickle
from transformers import AutoTokenizer
from src.utils.math_parser import extract_answer as math_extract_answer
from src.utils.prompts import ANSWER_PATTERN_MULTICHOICE
def load_responses_from_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        responses = pickle.load(f)
    model_name = "/".join(pkl_path.replace(".pkl","").split('_')[-2:])
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    assert all([i == int(responses[i].request_id) for i in range(len(responses))]), "Response request_ids do not match indices!"
    return [tokenizer.decode(resp.prompt_token_ids + resp.outputs[0].token_ids) for resp in responses], model_name
def extract_trace(output_text, model_name, full=False):
    """
    Given output text, extract the chain-of-thought if the tokenizer disambiguates it.
    
    Arguments:
    - output_text: str, full output text from the model.
    - model_name: str, name of the model to determine extraction method.
    - full: bool, default False. If True don't look for the end token.

    Returns the extracted reasoning trace as a string.
    Falls back to returning the full output_text if extraction fails.
    """
    try:
        if 'gpt-oss' in model_name.lower():
            return extract_trace_gpt(output_text, full)
        elif 'deepseek' in model_name.lower():
            return extract_trace_qwen(output_text, full)
        elif 'qwen' in model_name.lower() or 'qwq' in model_name.lower():
            return extract_trace_qwen(output_text, full)
        elif 'magistral' in model_name.lower():
            return extract_trace_mistral(output_text, full)
        elif 'gemma' in model_name.lower():
            return extract_trace_gemma(output_text, full)
        elif 'llama' in model_name.lower():
            return extract_trace_llama(output_text, full)
        elif "openreasoning" in model_name.lower():
            return extract_trace_qwen(output_text, full)
        elif 'phi' in model_name.lower():
            return extract_trace_phi(output_text, full)
        else:
            return output_text
    except (ValueError, IndexError) as e:
        return output_text
def extract_trace_llama(output_text, full = False):
    ass_start_tok = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    answer_start = output_text.index(ass_start_tok) + len(ass_start_tok)
    starts_thinking = "<think>"
    stops_thinking = "</think>"
    if starts_thinking in output_text[answer_start:]:
        answer_start = output_text.index(starts_thinking, answer_start) + len("<think>")
        if stops_thinking in output_text[answer_start:] and not full:
            answer_end = output_text.index(stops_thinking, answer_start)
            return output_text[answer_start:answer_end].replace(starts_thinking, "").replace(stops_thinking,"").strip()
        else:
            if "<|eot_id|>" not in output_text[answer_start:]:
                return output_text[answer_start:]
            else:
                answer_end = output_text.index("<|eot_id|>", answer_start)
                return output_text[answer_start:answer_end].replace(starts_thinking, "").replace(stops_thinking,"").strip()
    else:
        if "<|eot_id|>" not in output_text[answer_start:]:
            return output_text[answer_start:].replace(starts_thinking, "").replace(stops_thinking,"").strip()
        else:
            answer_end = output_text.index("<|eot_id|>", answer_start)
            return output_text[answer_start:answer_end]
def extract_trace_gemma(output_text, full=False):
    ass_start_tok = "<start_of_turn>model"
    ass_stop_tok = "<end_of_turn>"
    answer_start = output_text.index(ass_start_tok) + len(ass_start_tok)
    if ass_stop_tok not in output_text[answer_start:]:
        full_answer = output_text[answer_start:]
    else:
        answer_end = output_text.index(ass_stop_tok, answer_start)
        full_answer = output_text[answer_start:answer_end].replace(ass_start_tok, "").replace(ass_stop_tok, "").strip()
    return full_answer
def extract_trace_phi(output_text, full=True):
    ass_start_tok = "<|assistant|>"
    assert ass_start_tok in output_text
    return output_text[output_text.index(ass_start_tok):]
def extract_trace_mistral(output_text, full=False):
    starts_thinking = "[/INST]"
    stops_thinking = "</s>"
    if starts_thinking not in output_text:
        start = 0
    else:
        start = output_text.index(starts_thinking) + len(starts_thinking)
    if full or stops_thinking not in output_text[start:]:
        stop = len(output_text)
    else:
        stop = output_text.index(stops_thinking, start)
    return output_text[start:stop].replace(starts_thinking, "").replace(stops_thinking, "").strip()
def extract_trace_gpt(output_text, full=False):
    starts_thinking = "<|start|>assistant<|channel|>analysis<|message|>"
    stops_thinking = "<|end|>"
    channel_switch = "<|start|>assistant<|channel|>final<|message|>"
    returns = "<|return|>"
    if starts_thinking not in output_text:
        start = 0
    else:
        start = output_text.index(starts_thinking) + len(starts_thinking)
    if full or stops_thinking not in output_text[start:]:
        stop = len(output_text)
    else:
        stop = output_text.index(stops_thinking, start)
    return output_text[start:stop].replace(starts_thinking, "").replace(stops_thinking, "") \
            .replace(channel_switch, "").replace(returns, "").strip()
def extract_trace_qwen(output_text, full=False):
    ass_start_tok, ass_stop_tok = "<|im_start|>assistant\n", "<|im_end|>"
    starts_thinking = "<think>"
    stops_thinking = "</think>"
    if starts_thinking not in output_text:
        if ass_start_tok not in output_text:
            start = 0
        else:
            start = output_text.index(ass_start_tok) + len(ass_start_tok)
    else:
        start = output_text.index(starts_thinking) + len(starts_thinking)
    if full or stops_thinking not in output_text:
        stop = len(output_text)
    else:
        if stops_thinking in output_text[start:]:
            stop = output_text.index(stops_thinking, start)
        elif ass_stop_tok in output_text[start:]:
            stop = output_text.index(ass_stop_tok, start)
        else:
            stop = len(output_text)
    return output_text[start:stop].replace(starts_thinking, "").replace(stops_thinking, "").replace(ass_stop_tok, "").strip()
def extract_answer(output_text, dataset, model_name):
    """
    
    Calls dataset-specific answer extraction functions.

    Arguments:
    - output_text: str, full output text from the model.
    - dataset: str, name of the dataset to determine extraction method.
    - model_name: str, name of the model to clip generations for faulty responses.

    Returns the extracted answer as a string.

    """
    if 'math' in dataset.lower():
        return extract_answer_math(output_text, model_name)
    elif 'gpqa' in dataset.lower():
        return extract_answer_gpqa(output_text, model_name)
    elif 'connections' in dataset.lower():
        return extract_answer_connections(output_text, model_name)
    else:
        raise NotImplementedError(f"Unknown dataset for answer extraction: {dataset}.")
def extract_answer_math(output_text, model_name):
    assistant_only = extract_trace(output_text, model_name, full=True)
    return math_extract_answer(assistant_only)
def extract_answer_gpqa(output_text, model_name):
    assistant_only = extract_trace(output_text, model_name, full=True)
    matches = re.findall(ANSWER_PATTERN_MULTICHOICE, assistant_only)
    return matches[-1].upper() if matches else None
def extract_answer_connections(output_text, model_name):
    assistant_only = extract_trace(output_text, model_name, full=True)
    matches = re.findall(r"```json(.*?)```", assistant_only, re.DOTALL | re.IGNORECASE)
    raw_data = matches[-1].upper() if matches else None
    matches = re.findall(r"```json(.*?)```", assistant_only, re.DOTALL | re.IGNORECASE)
    raw_data = matches[-1].strip() if matches else None
    if raw_data is None:
        json_start = assistant_only.rfind('{')
        while json_start != -1:
            brace_count = 0
            json_end = -1
            for i in range(json_start, len(assistant_only)):
                if assistant_only[i] == '{':
                    brace_count += 1
                elif assistant_only[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i
                        break
            if json_end != -1:
                json_candidate = assistant_only[json_start:json_end+1]
                try:
                    test_data = json.loads(json_candidate)
                    if 'Groupings' in test_data or 'GROUPINGS' in {k.upper(): v for k, v in test_data.items()}:
                        raw_data = json_candidate
                        break
                except:
                    pass
            json_start = assistant_only.rfind('{', 0, json_start)
    try:
        if raw_data:
            raw_data_upper = raw_data.upper()
            data = json.loads(raw_data_upper)
            return {group["CATEGORY"]: group["WORDS"] for group in data["GROUPINGS"].values()}
        else:
            return None
    except Exception as e:
        print(f"Error extracting answer from connections: {e}")
        return None
