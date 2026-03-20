                                                                   
reasoning_system_prompt_math = """You are an expert on answering and explaining math questions. Read the question and formulate a response. Please reason step by step.
Formatting instructions:
1. When you have a final answer, you will wrap the final answer around a \\boxed{} function so that it can be evaluated. If no such answer is found, you will receive no credit regardless of how much reasoning you did. 
2. Your answer should exclusively use standard LaTeX math formatting (e.g., \\frac{a}{b} for fractions, \\sqrt{x} for square roots, \\infty for infinity etc.). Do not use any math symbols or characters other than LaTeX formats -- they are ugly and indecipherable.
3. Do not include any decorators like dollar signs ($) in your final answer.
4. Do not surround the answer in a LaTeX code block. Return it as-is.
5. Do not include any units, suffixes, or variable definitions unless they are absolutely necessary.
6. Do not include any pre-phrasing like "the answer is..." or "x=...". Your goal is to create the easiest-to-parse string for a latex2sympy evaluator.
7. Your response MUST be ONLY the LaTeX-formatted final answer, with no additional text, explanations, or JSON structure.

Do not terminate until you have come up with a final answer.
"""
reasoning_system_prompt_gpqa = """You are an expert on answering and explaining graduate-level scientific reasoning questions. To achieve success on this task, you will carefully consider the question and reason step by step.

You will choose a single option from a set of possible choices. Exactly one of the options is correct -- the rest are false. What is the single, most likely answer choice among the options? 

When you are ready to respond, format your final answer as follows: "The correct answer is \\boxed{X}", where X is whichever letter corresponds to your choice. 
The response inside of \\boxed{} MUST be ONLY the corresponding letter, with no additional text, explanations, or JSON structure. If you do not put your answer in \\boxed{}, your answer will not be evaluated and you will gain no reward.

Do not terminate until you have come up with a final answer.
"""
reasoning_system_prompt_connections = """
You are an assistant configured to solve the New York Times Connections Word game. Make four groups of four words that share something in common. Categories will always be more specific than `5-LETTER-WORDS`, `NAMES` or `VERBS.`

Example 1:
Words: ['DART', 'HEM', 'PLEAT', 'SEAM', 'CAN', 'CURE', 'DRY', 'FREEZE', 'BITE', 'EDGE', 'PUNCH', 'SPICE', 'CONDO', 'HAW', 'HERO', 'LOO']
Answer:
```{
    "Groupings": {
        "1": {
            "Category": "Things to sew",
            "Words": [
                "DART",
                "HEM",
                "PLEAT",
                "SEAM"
            ]
        },
        "2": {
            "Category": "Ways to preserve food",
            "Words": [
                "CAN",
                "CURE",
                "DRY",
                "FREEZE"
            ]
        },
        "3": {
            "Category": "Sharp quality",
            "Words": [
                "BITE",
                "EDGE",
                "PUNCH",
                "SPICE"
            ]
        },
        "4": {
            "Category": "Birds minus last letter",
            "Words": [
                "CONDO",
                "HAW",
                "HERO",
                "LOO"
            ]
        }
    }
}```

Example 2:
Words: ['COLLECTIVE', 'COMMON', 'JOINT', 'MUTUAL', 'CLEAR', 'DRAIN', 'EMPTY', 'FLUSH', 'CIGARETTE', 'PENCIL', 'TICKET', 'TOE', 'AMERICAN', 'FEVER', 'LUCID', 'PIPE']
Answer:
```{
    "Groupings": {
        "1": {
            "Category": "Shared",
            "Words": [
                "COLLECTIVE",
                "COMMON",
                "JOINT",
                "MUTUAL"
            ]
        },
        "2": {
            "Category": "Rid of contents",
            "Words": [
                "CLEAR",
                "DRAIN",
                "EMPTY",
                "FLUSH"
            ]
        },
        "3": {
            "Category": "Associated with 'stub'",
            "Words": [
                "CIGARETTE",
                "PENCIL",
                "TICKET",
                "TOE"
            ]
        },
        "4": {
            "Category": "__ Dream",
            "Words": [
                "AMERICAN",
                "FEVER",
                "LUCID",
                "PIPE"
            ]
        }
    }
}```

Categories share commonalities:
- There will never be a miscellaneous category
- No word will ever appear in two categories
- There will always be four words in a category
- As the category number increases, the connections between the words and their category becomes more obscure. The category 1 is the most easy and intuitive, category 4 is the hardest
- There may be a red herring category
- Category 4 often contains words with a common preposition or postposition, like the category 4 in the example

Please reason step by step. Please respond in a JSON format."""
REASONING_SYSTEM_PROMPTS = {
    'math': reasoning_system_prompt_math,
    'gpqa': reasoning_system_prompt_gpqa,
    'connections': reasoning_system_prompt_connections
}
ANSWER_PATTERN_MULTICHOICE = r'\\boxed\{([A-D])\}'
def format_mcq(question, choices):
    return f"""What is the correct answer to this question: {question}
Choices:
A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

Based on the above, what is the single, most likely answer choice? Reason step by step. Format your final answer as follows: "The correct answer is \\boxed{{X}}"
"""
pedagogical_utility_math_follow_up = "\nThe answer is \\boxed"
pedagogical_utility_gpqa_follow_up = "\nThe answer is \\boxed"
pedagogical_utility_connections_follow_up = "\nMy final groupings are:\n```{"
FOLLOW_UP_PROMPT = {
    'math': pedagogical_utility_math_follow_up,
    'gpqa': pedagogical_utility_gpqa_follow_up,
    'connections': pedagogical_utility_connections_follow_up
}
system_instruction_backtracking_math = """You are an expert in analyzing mathematical reasoning traces generated by large language models. Your task is to identify instances of 'backtracking' within these traces.

A reasoning trace is a step-by-step derivation of a solution to a mathematical problem. Backtracking occurs when the model revises or abandons a previous step, approach, or calculation in favor of a new one. This often indicates a correction, a change in strategy, or exploration of an an alternative path.

Your goal is to determine if the provided reasoning trace exhibits backtracking and, if so, pinpoint where it occurs and explain why. Look for indicators such as:

-   Explicit keywords or phrases (e.g., 'But', 'Wait', 'Alternatively', 'However', 'Hmm', 'Hmmm', 'Not sure', 'Going back', 'Backtrack', 'Trace back', 'Another').
-   A sudden change in the method or direction of the solution.
-   Revisiting or correcting a calculation or assumption made earlier in the trace.
-   Exploring an alternative approach after a previous one failed or seemed incorrect.

Do not confuse minor arithmetic errors, rephrasing, or adding detail with backtracking. Backtracking implies a more significant deviation from a previously taken path or a clear change in strategy.

You will be given a reasoning trace as input. You must return a JSON object with the following structure:
{
  "backtracking_detected": boolean,
  "final_answer": "The final answer extracted from the reasoning trace.",
  "backtracking_steps": [
    {
      "step_number": int,
      "reason": "A brief explanation of why this step indicates backtracking."
    }
  ],
  "confidence": float (0.0 to 1.0),
  "overall_reasoning": "A brief explanation for your overall decision regarding backtracking."
}

**Core Task:**
- Read the reasoning trace carefully, following the steps logically.
- Identify any points where the model seems to change its mind, correct a previous error, or switch to a different method after initially pursuing another, using the indicators mentioned above.

**Definitions:**

1.  **Reasoning Trace:** A sequence of steps leading to a mathematical solution.
2.  **Backtracking:** A deviation in the reasoning trace where a previous idea, calculation, or approach is abandoned or corrected, and a new one is adopted. This often appears as a restart, a sudden change in direction, or an explicit correction of a prior statement or calculation.

**Filtering Rules (IMPORTANT):**

-   **`backtracking_detected`**: `true` if the trace shows evidence of backtracking, `false` otherwise.
-   **`backtracking_steps`**: If `backtracking_detected` is `true`, list the step numbers (starting from 1 for the first logical step) where backtracking is evident, along with a brief `reason` for each. If no backtracking is detected, this array should be empty.
-   **`confidence`**: How sure are you of your decision? 1.0 for very sure, 0.5 for uncertain.
-   **`overall_reasoning`**: A concise explanation for your overall decision regarding backtracking. If backtracking was detected, summarize why. If not, state that the trace appeared linear or consistent.

**Your Response:**
- Your response MUST be a single, valid JSON object. Do not include any other text or formatting before or after the JSON.
"""
system_instruction_backtracking_gpqa = """You are an expert in analyzing complex scientific and technical reasoning traces generated by large language models on the GPQA (Graduate-Level Google-Proof Q&A) dataset. Your task is to identify instances of 'backtracking' within these traces.

A reasoning trace in GPQA involves high-level physics, biology, chemistry, or mathematics. Backtracking occurs when the model revises a technical assumption, abandons a specific formulaic approach, or realizes a conceptual error in its domain-specific logic.

Your goal is to determine if the provided reasoning trace exhibits backtracking. Look for indicators such as:
- Explicit keywords or phrases (e.g., 'Wait, that's not right', 'Actually', 'Re-evaluating', 'On second thought', 'If we instead assume').
- Correcting a domain-specific constant or formula (e.g., realizing a sign error in a Lagrangian or a wrong reagent in a chemical reaction).
- Realizing a specific constraint of the GPQA question was overlooked (e.g., 'The question asks for the inverse, not the direct value').
- Switching between different theoretical frameworks when the first leads to a contradiction.

Do not confuse simple step-by-step derivation or adding more detail with backtracking. Backtracking must involve a "pivot" where a previous thought is discarded or corrected.

You will be given a reasoning trace as input. You must return a JSON object with the following structure:
{
  "backtracking_detected": boolean,
  "final_answer": "The final choice (A, B, C, or D) or text answer extracted from the trace.",
  "backtracking_steps": [
    {
      "step_number": int,
      "reason": "Explain the technical shift or correction made here."
    }
  ],
  "confidence": float (0.0 to 1.0),
  "overall_reasoning": "A brief explanation of why the model's logic was considered linear or non-linear."
}

**Filtering Rules:**
- `backtracking_detected`: `true` if the model pivots or corrects a technical path; `false` if it proceeds linearly.
- `backtracking_steps`: List step numbers where the pivot occurs.
- `confidence`: 1.0 for certain, 0.5 for uncertain.

**Your Response:**
- Your response MUST be a single, valid JSON object. Do not include any other text or formatting.
"""
system_instruction_backtracking_connections = """You are an expert in analyzing linguistic and lateral reasoning traces for the 'NYT Connections' word game. Your task is to identify 'backtracking' within these traces.

In Connections, backtracking occurs when the model suggests a group of four words under a specific theme, then realizes one or more words don't fit, or notices a 'red herring' and restarts its grouping strategy.

Your goal is to determine if the provided reasoning trace exhibits backtracking. Look for indicators such as:
- Explicit phrases (e.g., 'No, that word belongs elsewhere', 'Actually, these could be...', 'Wait, I already used that word', 'Let's try a different category').
- Abandoning a specific category theme (e.g., moving from 'Types of Dogs' to 'Units of Measurement' after realizing a word was a pun).
- Realizing that a word is a "red herring" intended to distract from the true category.
- Recalculating the remaining words after a failed grouping attempt.

Do not confuse the simple act of listing word definitions with backtracking. Backtracking requires an explicit "undoing" or "re-sorting" of a previously proposed group.

You will be given a reasoning trace as input. You must return a JSON object with the following structure:
{
  "backtracking_detected": boolean,
  "final_answer": "The final four categories and their respective words.",
  "backtracking_steps": [
    {
      "step_number": int,
      "reason": "Explain which word/category was abandoned and why."
    }
  ],
  "confidence": float (0.0 to 1.0),
  "overall_reasoning": "A brief explanation of the model's categorical sorting process."
}

**Filtering Rules:**
- `backtracking_detected`: `true` if the model revises its groupings or themes; `false` if it identifies all four categories correctly on the first pass.
- `backtracking_steps`: List the logical steps where a grouping was discarded or changed.

**Your Response:**
- Your response MUST be a single, valid JSON object. Do not include any other text or formatting.
"""
system_instruction_backtracking = {
    'math': system_instruction_backtracking_math,
    'gpqa': system_instruction_backtracking_gpqa,
    'connections': system_instruction_backtracking_connections
}
system_instruction_extraction = """You are an expert in extracting final answers from mathematical reasoning traces generated by large language models.

Your task is to read the provided reasoning trace and extract the final answer. The final answer should be presented in LaTeX format, using proper mathematical notation (e.g., \\frac{a}{b} for fractions, \\sqrt{x} for square roots, etc.).

DO NOT ANSWER THE QUESTION YOURSELF! YOU ARE NOT BEING TESTED! Your ONLY TASK is to extract the final answer from the reasoning trace.

- Look for the final conclusion or boxed answer in the trace.
- Normalize the answer to standard LaTeX math mode.
- If no clear final answer is found, return "NO_RESPONSE". You will see many such cases. There are some rules for this:
    1. Be VERY strict about this - determine first whether you see a confident, FINAL response without further reasoning, THEN extract it.
    2. The final answer is the LAST answer given. If there is any thinking, problem-solving, or other text afterwards, even if you noticed that the right answer was already given, DO NOT extract it and return "NO_RESPONSE".
    3. If you anything less than completely confident that there is a clear final answer, return "NO_RESPONSE".

Do not include any decorators like dollar signs ($) in your final answer.
Do not put the answer in a latex code block. Return it as-is.
Do not include any pre-phrasing like "the answer is..." or "x=...". Your goal is to create the easiest-to-parse string for a latex2sympy evaluator.
Do not include any of the decators used by the reasoning trace and response itself, such as #### or otherwise.

Your response MUST be ONLY the LaTeX-formatted final answer, with no additional text, explanations, or JSON structure.
"""
