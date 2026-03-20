"""
The logic in this file largely borrows from Qwen2.5-Math codebase at https://github.com/QwenLM/Qwen2.5-Math:
"""
import re
from math import isclose
from word2number import w2n
def convert_word_number(text: str) -> str:
    try:
        text = str(w2n.word_to_num(text))
    except Exception:
        pass
    return text
def _fix_fracs(string):
    substrs = string.split('\\frac')
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += '\\frac'
            if len(substr) > 0 and substr[0] == '{':
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != '{':
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}{' + b + '}' + post_substr
                    else:
                        new_str += '{' + a + '}{' + b + '}'
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}' + b + post_substr
                    else:
                        new_str += '{' + a + '}' + b
    string = new_str
    return string
def _fix_a_slash_b(string):
    if len(string.split('/')) != 2:
        return string
    a = string.split('/')[0]
    b = string.split('/')[1]
    try:
        if 'sqrt' not in a:
            a = int(a)
        if 'sqrt' not in b:
            b = int(b)
        assert string == '{}/{}'.format(a, b)
        new_string = '\\frac{' + str(a) + '}{' + str(b) + '}'
        return new_string
    except Exception:
        return string
def _fix_sqrt(string):
    _string = re.sub(r'\\sqrt(\w+)', r'\\sqrt{\1}', string)
    return _string
def strip_answer_string(string):
    string = str(string).strip()
    string = string.replace('\n', '')
    string = string.rstrip('.')
    string = string.replace('\\!', '')
    string = re.sub(r'\\begin\{array\}\{.*?\}', r'\\begin{pmatrix}', string)
    string = re.sub(r'\\end\{array\}', r'\\end{pmatrix}', string)
    string = string.replace('bmatrix', 'pmatrix')
    string = string.replace('tfrac', 'frac')
    string = string.replace('dfrac', 'frac')
    string = (string.replace('\\neq', '\\ne').replace('\\leq', '\\le').replace('\\geq', '\\ge'))
    string = string.replace('\\left', '')
    string = string.replace('\\right', '')
    string = string.replace('\\{', '{')
    string = string.replace('\\}', '}')
    def replace_match(match):
        word = match.group(1).lower()
        if convert_word_number(word) == word:
            return match.group(0)
        else:
            return convert_word_number(word)
    string = re.sub(r'\\text\{([a-zA-Z]+)\}', replace_match, string)
    string = re.sub(r'(cm|inches)\}\^2', r'\1}', string)
    _string = re.sub(r'\\text{.*?}$', '', string).strip()
    if _string != '' and _string != string:
        string = _string
    string = string.replace('^{\\circ}', '')
    string = string.replace('^\\circ', '')
    string = string.replace('\\$', '')
    string = string.replace('$', '')
    string = string.replace('\\(', '').replace('\\)', '')
    string = convert_word_number(string)
    string = re.sub(r'\\text\{(.*?)\}', r'\1', string)
    for key in ['x=', 'y=', 'z=', 'x\\in', 'y\\in', 'z\\in', 'x\\to', 'y\\to', 'z\\to']:
        string = string.replace(key, '')
    string = string.replace('\\emptyset', r'{}')
    string = string.replace('(-\\infty,\\infty)', '\\mathbb{R}')
    string = string.replace('\\%', '')
    string = string.replace('%', '')
    string = string.replace(' .', ' 0.')
    string = string.replace('{.', '{0.')
    if (
        string.startswith('{') and string.endswith('}') and string.isalnum()
        or string.startswith('(') and string.endswith(')') and string.isalnum()
        or string.startswith('[') and string.endswith(']') and string.isalnum()
    ):
        string = string[1:-1]
    string = string.replace('infinity', '\\infty')
    if '\\infty' not in string:
        string = string.replace('inf', '\\infty')
    string = string.replace('+\\inity', '\\infty')
    string = string.replace('and', '')
    string = string.replace('\\mathbf', '')
    string = re.sub(r'\\mbox{.*?}', '', string)
    string.replace("'", '')
    string.replace('"', '')
    if 'j' in string and 'i' not in string:
        string = string.replace('j', 'i')
    string = re.sub(r'(\d+)\.0*([^\d])', r'\1\2', string)
    string = re.sub(r'(\d+)\.0*$', r'\1', string)
    if len(string) == 0:
        return string
    if string[0] == '.':
        string = '0' + string
    if len(string.split('=')) == 2:
        if len(string.split('=')[0]) <= 2:
            string = string.split('=')[1]
    string = _fix_sqrt(string)
    string = string.replace(' ', '')
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)
    string = re.sub(r'\\(?=\-?\d+(\\|\)|,|\]|$))', '', string)
    string = re.sub(r'thgrade$', '', string)
    if re.fullmatch(r'\s*-?\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*', string):
        string = string.replace(',', '')
    if re.fullmatch(r'(\s*-?\d+\s*,)*\s*-?\d+\s*', string):
        try:
            integer_list = list(map(int, string.split(',')))
        except Exception:
            integer_list = list(map(int, '-1,-1'.split(',')))
        sorted_list = sorted(integer_list)
        string = ','.join(map(str, sorted_list))
    return string
def extract_answer(pred_str, use_last_number=True):
    pred_str = pred_str.replace('\u043a\u0438', '')
    if 'final answer is $' in pred_str and '$. I hope' in pred_str:
        tmp = pred_str.split('final answer is $', 1)[1]
        pred = tmp.split('$. I hope', 1)[0].strip()
    elif 'boxed' in pred_str:
        ans = pred_str.split('boxed')[-1]
        if len(ans) == 0:
            return ''
        elif ans[0] == '{':
            stack = 1
            a = ''
            for c in ans[1:]:
                if c == '{':
                    stack += 1
                    a += c
                elif c == '}':
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        pred = a
    elif 'he answer is' in pred_str:
        pred = pred_str.split('he answer is')[-1].strip()
    elif 'final answer is' in pred_str:
        pred = pred_str.split('final answer is')[-1].strip()
    elif '答案是' in pred_str:
        pred = pred_str.split('答案是')[1].strip().split('\n\n')[0].strip()
    elif 'ANSWER:' in pred_str:
        pred = pred_str.split('ANSWER:')[-1].strip()
    else:
        if use_last_number:
            pattern = r'-?\d*\.?\d+'
            pred = re.findall(pattern, pred_str.replace(',', ''))
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ''
        else:
            pred = ''
    pred = re.sub(r'\n\s*', '', pred)
    if pred != '' and pred[0] == ':':
        pred = pred[1:]
    if pred != '' and pred[-1] == '.':
        pred = pred[:-1]
    if pred != '' and pred[-1] == '/':
        pred = pred[:-1]
    pred = strip_answer_string(pred)
    return pred
def extract_answer_idx(pred_str, use_last_number=True):
    """Extract answer and return (answer, start_index, end_index) where indices point to 
    the answer location in the ORIGINAL pred_str BEFORE any strip_answer_string cleaning."""
    original_pred_str = pred_str
    pred_str = pred_str.replace('\u043a\u0438', '')
    raw_answer_text = None
    raw_start_idx = -1
    raw_end_idx = -1
    if 'final answer is $' in pred_str and '$. I hope' in pred_str:
        marker = 'final answer is $'
        marker_pos = pred_str.index(marker)
        tmp_start = marker_pos + len(marker)
        tmp = pred_str[tmp_start:]
        raw_answer_text = tmp.split('$. I hope', 1)[0].strip()
        raw_start_idx = pred_str.index(raw_answer_text, tmp_start)
        raw_end_idx = raw_start_idx + len(raw_answer_text)
    elif 'boxed' in pred_str:
        boxed_pos = pred_str.rindex('boxed')
        ans = pred_str[boxed_pos + len('boxed'):]
        if len(ans) == 0:
            return ('', -1, -1)
        elif ans[0] == '{':
            stack = 1
            raw_answer_text = ''
            char_idx = 1
            for c in ans[1:]:
                if c == '{':
                    stack += 1
                    raw_answer_text += c
                elif c == '}':
                    stack -= 1
                    if stack == 0:
                        break
                    raw_answer_text += c
                else:
                    raw_answer_text += c
                char_idx += 1
            raw_start_idx = boxed_pos + len('boxed') + 1
            raw_end_idx = raw_start_idx + len(raw_answer_text)
        else:
            raw_answer_text = ans.split('$')[0].strip()
            strip_offset = ans.index(raw_answer_text) if raw_answer_text in ans else 0
            raw_start_idx = boxed_pos + len('boxed') + strip_offset
            raw_end_idx = raw_start_idx + len(raw_answer_text)
    elif 'he answer is' in pred_str:
        marker = 'he answer is'
        marker_pos = pred_str.rindex(marker)
        after_marker = pred_str[marker_pos + len(marker):]
        raw_answer_text = after_marker.strip()
        raw_start_idx = pred_str.index(raw_answer_text, marker_pos + len(marker))
        raw_end_idx = raw_start_idx + len(raw_answer_text)
    elif 'final answer is' in pred_str:
        marker = 'final answer is'
        marker_pos = pred_str.rindex(marker)
        after_marker = pred_str[marker_pos + len(marker):]
        raw_answer_text = after_marker.strip()
        raw_start_idx = pred_str.index(raw_answer_text, marker_pos + len(marker))
        raw_end_idx = raw_start_idx + len(raw_answer_text)
    elif '答案是' in pred_str:
        marker = '答案是'
        marker_pos = pred_str.rindex(marker)
        after_marker = pred_str[marker_pos + len(marker):]
        raw_answer_text = after_marker.split('\n\n')[0].strip()
        raw_start_idx = pred_str.index(raw_answer_text, marker_pos + len(marker))
        raw_end_idx = raw_start_idx + len(raw_answer_text)
    elif 'ANSWER:' in pred_str:
        marker = 'ANSWER:'
        marker_pos = pred_str.rindex(marker)
        after_marker = pred_str[marker_pos + len(marker):]
        raw_answer_text = after_marker.strip()
        raw_start_idx = pred_str.index(raw_answer_text, marker_pos + len(marker))
        raw_end_idx = raw_start_idx + len(raw_answer_text)
    else:
        if use_last_number:
            pattern = r'-?\d*\.?\d+'
            matches = list(re.finditer(pattern, pred_str))
            if len(matches) >= 1:
                last_match = matches[-1]
                raw_answer_text = last_match.group()
                raw_start_idx = last_match.start()
                raw_end_idx = last_match.end()
            else:
                raw_answer_text = ''
                raw_start_idx = -1
                raw_end_idx = -1
        else:
            raw_answer_text = ''
            raw_start_idx = -1
            raw_end_idx = -1
    if raw_answer_text is None:
        cleaned_answer = ''
    else:
        cleaned_answer = raw_answer_text
        cleaned_answer = re.sub(r'\n\s*', '', cleaned_answer)
        if cleaned_answer != '' and cleaned_answer[0] == ':':
            cleaned_answer = cleaned_answer[1:]
        if cleaned_answer != '' and cleaned_answer[-1] == '.':
            cleaned_answer = cleaned_answer[:-1]
        if cleaned_answer != '' and cleaned_answer[-1] == '/':
            cleaned_answer = cleaned_answer[:-1]
        cleaned_answer = strip_answer_string(cleaned_answer)
    return cleaned_answer, raw_start_idx, raw_end_idx
def choice_answer_clean(pred: str):
    pred = pred.strip('\n').rstrip('.').rstrip('/').strip(' ').lstrip(':')
    tmp = re.findall(r'\b(A|B|C|D|E)\b', pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip('.')]
    pred = pred[-1]
    pred = pred.rstrip('.').rstrip('/')
    return pred
def parse_digits(num):
    num = re.sub(',', '', str(num))
    try:
        return float(num)
    except Exception:
        if num.endswith('%'):
            num = num[:-1]
            if num.endswith('\\'):
                num = num[:-1]
            try:
                return float(num) / 100
            except Exception:
                pass
    return None
def is_digit(num):
    return parse_digits(num) is not None
def str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r'\{.*,.*\}', input_str)
    pmatrix_list = []
    for m in matrix_str:
        m = m.strip('{}')
        pmatrix = r'\begin{pmatrix}' + m.replace(',', '\\') + r'\end{pmatrix}'
        pmatrix_list.append(pmatrix)
    return ', '.join(pmatrix_list)
def math_equal(
    prediction,
    reference,
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    if prediction is None or reference is None:
        return False
    if str(prediction.strip().lower()) == str(reference.strip().lower()):
        return True
    if (reference in ['A', 'B', 'C', 'D', 'E'] and choice_answer_clean(prediction) == reference):
        return True
    try:
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(prediction, item):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except Exception:
        pass
    if not prediction and prediction not in [0, False]:
        return False
    reference = str(reference).strip()
    prediction = str(prediction).strip()
    if 'pmatrix' in prediction and 'pmatrix' not in reference:
        reference = str_to_pmatrix(reference)
    pred_str, ref_str = prediction, reference
    if (prediction.startswith('[') and prediction.endswith(']') and not reference.startswith('(')
        ) or (prediction.startswith('(') and prediction.endswith(')') and not reference.startswith('[')):
        pred_str = pred_str.strip('[]()')
        ref_str = ref_str.strip('[]()')
    for s in ['{', '}', '(', ')']:
        ref_str = ref_str.replace(s, '')
        pred_str = pred_str.replace(s, '')
    if pred_str.lower() == ref_str.lower():
        return True
    if (
        re.match(r'(\(|\[).+(\)|\])', prediction) is not None
        and re.match(r'(\(|\[).+(\)|\])', reference) is not None
    ):
        pred_parts = prediction[1:-1].split(',')
        ref_parts = reference[1:-1].split(',')
        if len(pred_parts) == len(ref_parts):
            if all([
                math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close) for i in range(len(pred_parts))
            ]):
                return True
    if ((prediction.startswith('\\begin{pmatrix}') or prediction.startswith('\\begin{bmatrix}'))
        and (prediction.endswith('\\end{pmatrix}') or prediction.endswith('\\end{bmatrix}'))
        and (reference.startswith('\\begin{pmatrix}') or reference.startswith('\\begin{bmatrix}'))
        and (reference.endswith('\\end{pmatrix}') or reference.endswith('\\end{bmatrix}'))):
        pred_lines = [
            line.strip()
            for line in prediction[len('\\begin{pmatrix}'):-len('\\end{pmatrix}')].split('\\\\')
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[len('\\begin{pmatrix}'):-len('\\end{pmatrix}')].split('\\\\')
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split('&')
                ref_parts = ref_line.split('&')
                if len(pred_parts) == len(ref_parts):
                    if not all([
                        math_equal(
                            pred_parts[i],
                            ref_parts[i],
                            include_percentage,
                            is_close,
                        ) for i in range(len(pred_parts))
                    ]):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True
    if prediction.count('=') == 1 and reference.count('=') == 1:
        pred = prediction.split('=')
        pred = f'{pred[0].strip()} - ({pred[1].strip()})'
        ref = reference.split('=')
        ref = f'{ref[0].strip()} - ({ref[1].strip()})'
        if symbolic_equal(pred, ref) or symbolic_equal(f'-({pred})', ref):
            return True
    elif (prediction.count('=') == 1 and len(prediction.split('=')[0].strip()) <= 2 and '=' not in reference):
        if math_equal(prediction.split('=')[1], reference, include_percentage, is_close):
            return True
    elif (reference.count('=') == 1 and len(reference.split('=')[0].strip()) <= 2 and '=' not in prediction):
        if math_equal(prediction, reference.split('=')[1], include_percentage, is_close):
            return True
    if symbolic_equal(prediction, reference):
        return True
    return False
def numeric_equal(prediction: float, reference: float):
    return isclose(reference, prediction, rel_tol=1e-4)
def symbolic_equal(a, b):
    try:
        from sympy import N, simplify
        from sympy.parsing.latex import parse_latex
        from sympy.parsing.sympy_parser import parse_expr
    except Exception as e:
        print("Warning: sympy not installed, symbolic_equal will always return False.")
        print(e)
        parse_latex = None
        parse_expr = None
        N = None
        simplify = None
    try:
        from latex2sympy2_extended import latex2sympy
    except Exception as e:
        print("Warning: latex2sympy2_extended not installed, symbolic_equal will always return False.")
        print(e)
        latex2sympy = None
    def _parse(s):
        parsers = []
        if parse_latex is not None:
            parsers.append(parse_latex)
        if parse_expr is not None:
            parsers.append(parse_expr)
        if latex2sympy is not None:
            parsers.append(latex2sympy)
        for f in parsers:
            try:
                return f(s.replace('\\\\', '\\'))
            except Exception:
                try:
                    return f(s)
                except Exception:
                    pass
        return s
    a = _parse(a)
    b = _parse(b)
    try:
        if str(a) == str(b) or a == b:
            return True
    except Exception:
        pass
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except Exception:
        pass
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except Exception:
        pass
    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except Exception:
        pass
    try:
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except Exception:
        pass
    return False
def warmup_math_grader():
    """
    Warmup function to trigger lazy imports of heavy symbolic libraries.
    Call this once before using multiprocessing to avoid timeout on first grading call.
    """
    _ = math_equal('1', '1')
    _ = math_equal('\\frac{1}{2}', '0.5')
    _ = math_equal('\\sqrt{2}', '1.414', is_close=True)
if __name__ == '__main__':
    print(math_equal('\n\\boxed{70,\\!000}\n', '70000'))
    print(extract_answer('The answer is \\boxed{70,\\!000}'))
    print(strip_answer_string(extract_answer('The answer is \\boxed{70,\\!000}')))
    print(math_equal(extract_answer('The answer is \\boxed{70,\\!000}'), '70000'))
