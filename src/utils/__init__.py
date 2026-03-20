
def load_dataset(dataset_name: str):
    """
    Dataset loader for supported datasets.
    """
    import datasets
    if dataset_name == 'math':
        dataset = datasets.load_from_disk("math_consolidated_test")
    elif dataset_name == 'gpqa':
        dataset = datasets.load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
    elif dataset_name == 'connections':
        dataset = datasets.load_dataset("tm21cy/NYT-Connections", split="train")
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")
    return dataset
def get_question_answer(dataset_name, item):
    """
    Given a dataset and an element, return the question prompt and (a) ground truth answer.
    Only supports 'math', 'gpqa', and 'connections' for now.

    """
    from src.utils import math_parser
    from src.utils.prompts import format_mcq
    import random
    if dataset_name == 'math':
        return item['problem'], math_parser.extract_answer(item['solution'])
    elif dataset_name == 'gpqa':
        question = item['Question']
        choices = [
            item["Correct Answer"],
            item["Incorrect Answer 1"],
            item["Incorrect Answer 2"],
            item["Incorrect Answer 3"],
        ]
        random.shuffle(choices)
        correct_letter = "ABCD"[choices.index(item["Correct Answer"])]
        answer = correct_letter
        return f"{format_mcq(question, choices)}", answer
    elif dataset_name == 'connections':
        question = f"Today's list of words: {item['words']}"
        answer = {
            item['answerDescription']: item['words']
            for item in item['answers']
        }   
        return question, answer
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")
