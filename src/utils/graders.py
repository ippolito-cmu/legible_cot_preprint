from src.utils.extractors import *
import src.utils.math_parser as math_parser
def warmup_grader():
    """Warmup grader to trigger lazy imports before multiprocessing"""
    math_parser.warmup_math_grader()
def grade_answer(dataset, extracted, ground_truth):
    if 'math' in dataset.lower():
        return 1 if math_parser.math_equal(extracted, ground_truth) else 0
    elif 'gpqa' in dataset.lower():
        return 1 if extracted == ground_truth else 0
    elif 'connections' in dataset.lower():
        if extracted is None or ground_truth is None:
            return 0
        return grade_answer_connections(extracted, ground_truth)
    else:
        raise NotImplementedError(f"Unknown dataset for grading: {dataset}")
def grade_answer_connections(extracted, ground_truth):
    import json
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            return 0
    if isinstance(extracted, str):
        try:
            extracted = json.loads(extracted)
        except json.JSONDecodeError:
            return 0
    if not isinstance(extracted, dict) or not isinstance(ground_truth, dict):
        return 0
    ground_truth_parsed = [value for _, value in ground_truth.items()]
    extracted_parsed = [value for _, value in extracted.items()]
    gt_set = set(frozenset(group) for group in ground_truth_parsed)
    return len([group for group in extracted_parsed if frozenset(group) in gt_set])
