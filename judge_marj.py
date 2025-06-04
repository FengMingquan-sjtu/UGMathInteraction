import sympy as sp
from sympy import simplify, Eq, sympify, Pow, N, Mul, trigsimp, S
from sympy.parsing.latex import parse_latex
import re
import signal
import math
import random
from sympy.calculus.util import continuous_domain
import numpy as np
from openai import OpenAI, RateLimitError
import time
from utils import *
import httpx
random.seed(123)

EXCLUDE_TYPE = ["UOL", "OL"]

def initialize_client():
    global client
    httpx_client = httpx.Client(verify=False)
    #os.environ["OPENAI_BASE_URL"] = ""
    #os.environ["OPENAI_API_KEY"] = ""

    client = OpenAI(http_client=httpx_client)

class Judger:
    def __init__(self, strict_extract = False, judge_model="gpt-4.1"):
        # TODO: add strict_extract as args in generate.py or evaluate.py
        self.judgment_methods = {
            "UOL": self.judge_unordered_list,
            "OL": self.judge_ordered_list,
            "INT": self.judge_interval,
            "TF": self.judge_TF,
            "EX": self.judge_expression,
            "EQ": self.judge_equation,
            "OE": self.judge_extract_match, # open-ended
            "MCM": self.judge_MC_multiple,
            "MCS": self.judge_MC_single,
            "NV": self.judge_single_numerical_value,
        }
        self.pi = parse_latex("\\pi")
        self.precision = 1e-8
        self.num_samples = 100 # number of numbers sampled from domain of definition each time
        self.num_times = 3 # times repeated to evaluate expression if it has variables
        self.strict_extract = strict_extract
        self.judge_model = judge_model

    def normalize_answer(self, final_answer):
        # TODO: add other normalize answer pattern
        special_signal_map = {
            "\\left": "",
            "\\right": "",
            "∶": ":",
            "，": ",",
            "$": "",
            "\\approx": "=",
            "\\simeq": "=",
            "\\sim": "=",
            "^\\prime": "'",
            "^{\\prime}": "'",
            "^\\circ": "",
            "%": "",
        }
        for signal in special_signal_map:
            final_answer = final_answer.replace(signal, special_signal_map[signal])
        final_answer = re.sub(r'\\(?:mathrm|mathbf)\{~?([^}]*)\}', '\\1', final_answer)
        final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
        final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
        final_answer = re.sub(
            r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
        final_answer = re.sub(
            r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
        final_answer = final_answer.strip()
        final_answer = final_answer.strip("$")
        final_answer = final_answer.strip()
        return final_answer
    
    def norm_ans_str(self, ans: str, ans_type: str) -> str:
        """Normalize answer string for **all kinds** of answers."""

        # do not change for MCS and MCM
        if ans_type in ["MCS", "MCM", "OE"]:
            return ans
        if len(ans) == 0:
            return ans
        if ans[0] == '(' and ans[-1] == ')' and ans_type in ["OL", "UOL"]:
            ans_list = self.split_by_comma(ans.strip("()"))
        elif ans[0] == '{' and ans[-1] == '}':
            ans_list = self.split_by_comma(ans.strip("{}"))
        else:
            ans_list = self.split_by_comma(ans)
        new_ans_list = []

        for ans_i in ans_list:
            ans_item = str(ans_i)
            ans_item = ans_item.replace("\n", "")
            ans_item = ans_item.strip()

            # remove impropriate trailing punctuations
            ans_item = self.clean(ans_item)

            # bool
            if ans_type == "TF" or ans_type == "OL" or ans_type == "UOL":
                ans_bool = norm_str2bool(ans_item)
                if ans_bool is not None:
                    new_ans_list.append(str(ans_bool))
                    continue

            # weekdays
            ans_weekday = norm_str2weekday(ans_item)
            if ans_weekday is not None:
                new_ans_list.append(str(ans_weekday))
                continue

            # math normalize
            ans_item = self.norm_math_str(ans_item)
            new_ans_list.append(ans_item)

        assert len(ans_list) == len(new_ans_list)
        if len(new_ans_list) == 1:
            return new_ans_list[0]
        return "(" + ", ".join(new_ans_list) + ")"
        
    def eq(self, ref: str, ans: str) -> bool:
        """Check if reference answer and prediction answer are **literally** equal."""
        return ref == ans

    def norm_pm(self, s: str) -> str:
        """Replaces the LaTeX symbols '$1\\pm$2' or '$1\\mp$2' with '$1-$2,$1+$2'."""

        def replace_pm(match):
            # Extracts the first and second parts of the match.
            first_part, second_part = match.groups()
            # Creates the replacement string as specified.
            return f"{first_part}-{second_part},{first_part}+{second_part}"

        _s = self.remove_out_paren(s)
        # Define the pattern that matches '$1\\pm$2' or '$1\\mp$2'.
        # We use non-greedy matching (.*?) to capture the parts before and after \pm or \mp.
        # The pattern is corrected to include the '$' signs and to capture the expressions correctly.
        pattern = r"([\w\.\\{}\+\-\*\^]+?)(?:\\pm|\\mp)([\w\.\\{}\+\-\*\^]+)"

        if re.search(pattern, _s):
            # Use re.sub to replace all occurrences of the pattern in the input string.
            return re.sub(pattern, replace_pm, _s)
        else:
            return s
    
    def extract_set(self, norm_s: str) -> list[str]:
        clean_s = self.remove_out_paren(norm_s)
        ele_strs = clean_s.replace("or", ",").split(",")
        ele_strs = [s.strip() for s in ele_strs]

        # ele_strs.sort()
        # return ele_strs

        merged_strs = []
        for i in range(len(ele_strs)):
            s_i = ele_strs[i]
            existing = False
            for j in range(i):
                s_j = ele_strs[j]
                if self.eq(s_i, s_j):
                    existing = True
                    break
            if not existing:
                merged_strs.append(s_i)

        merged_strs.sort()

        return merged_strs

    def remove_out_paren(self, s: str) -> str:
        """Remove until there are no parentheses outside."""
        done = False
        while not done:
            done = True
            for left, _ in PAREN_MAP.items():
                len_paren = len(left)
                i_l, i_r = self.index_first_paren_pair(s, left)
                if i_l == 0 and i_r == len(s) - len_paren:
                    s = s[len_paren:-len_paren]
                    done = False
        return s
    
    def remove_first_paren_pair(
        self,
        s: str,
        l: str,  # Left parenthesis
    ) -> str:
        i_l, i_r = self.index_first_paren_pair(s, l)
        if i_l != -1 and i_r != -1:
            len_paren = len(l)
            s = s[:i_l] + s[i_l + len_paren : i_r] + s[i_r + len_paren :]

        return s
    
    def remove_latex_cmd(self, s: str, cmd: str) -> str:
        try:
            cmd_idx = s.index(cmd)
        except ValueError:
            return s

        pfx = s[:cmd_idx].strip()
        sfx = s[cmd_idx + len(cmd) :].strip()

        if len(sfx) > 0 and sfx[0] == "{":  # Common command
            sfx = self.remove_first_paren_pair(sfx, "{")
        elif len(pfx) > 0 and pfx[-1] == "{":  # Declaration command
            left_idx_in_sfx = sfx.find("}")
            if left_idx_in_sfx != -1:
                pfx = pfx[:-1]
                sfx = sfx[:left_idx_in_sfx] + sfx[left_idx_in_sfx + 1 :]
        else:  # Indepedent command
            pass

        return pfx + sfx
    
    def norm_basic_fn(self, s: str) -> str:
        """Avoid potential LaTex errors caused by removing spaces:
        - \\{fn}[a-z] : followed by some letter without middle spaces
        - \\{fn}^{pow}{expr}

        Returns
        -------
        str
            Normalized format of basic function expression: \\{fn}^{{pow}}{{expr}}
        """
        # \2 matches \d+ without {} around, if there has been {}, there is no need to normalize
        # Existing nude power, i.e. ^<pow_d+>
        s = re.sub(rf"\\?({'|'.join(BASIC_FN_NAMES)})\^(\d+)", r"\\\1^{\2}", s)
        # No power
        s = re.sub(rf"\\?({'|'.join(BASIC_FN_NAMES)})(?!\^)", r"\\\1^{1}", s)
        return s


    def index_first_paren_pair(self, s: str, l: str) -> tuple[int, int]:
        r = PAREN_MAP[l]
        try:
            i_l = s.index(l)
        except ValueError:
            return -1, -1
        len_paren = len(l)

        depth = 0
        i_r = -1
        for i_c in range(i_l, len(s)):
            if s[i_c : i_c + len_paren] == l:
                depth -= 1
            elif s[i_c : i_c + len_paren] == r:
                depth += 1
            if depth == 0:
                i_r = i_c
                break

        return i_l, i_r
    
    def norm_math_str(self, string: str):

        string = str(string).strip()
        string = self.clean(string)

        # Simple removals
        for rm_str in SIMPLE_RM_STRS:
            string = string.replace(rm_str, "")

        # Simple replacements
        for k, v in SIMPLE_REPLACE_MAP.items():
            string = string.replace(k, v)
        if "\\infty" not in string:
            string = string.replace("inf", "\\infty")

        # Remove spaces after all space-related operations
        string = string.replace(" ", "")

        for latex_cmd in LATEX_CMDS:
            string = self.remove_latex_cmd(string, latex_cmd)

        for env in LATEX_FMT_ENVS + LATEX_LIST_ENVS:
            string = rm_latex_env(string, env)

        # Normalize local expressions
        string = norm_deg(string)  # Normalize degrees
        # convert inverse functions
        string = fix_inv_func(string)
        string = re.sub(
            rf"(?<!\\)(pi\b|{'|'.join(BASIC_FN_NAMES)})", r"\\\1", string
        )  # Fix backslashes
        string = self.norm_basic_fn(string)  # Normalize basic functions

        # Normalize matrix and array
        string = re.sub(r"{[a-z]?matrix}", r"{array}", string)
        string = re.sub(r"\\begin{array}{[lcr]*}", r"\\begin{array}{}", string)
        # NOTE: the substituion str should alse obey the regex syntax, like r"\\begin{array}"
        if "\\begin{array}" not in string:
            string = string.replace("\\\\", "")

        # i, j
        if "j" in string and "i" not in string:
            string = string.replace("j", "i")

        # replace a.000b where b is not number or b is end, with ab, use regex
        string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
        string = re.sub(r"(\d+)\.0+$", r"\1", string)

        # remove units
        for unit in UNITS:
            string = re.sub(f"([-\d\.\*\^{{}}]+){unit}e?s?$", "\\1", string)

        # Check if empty before splitting
        # if empty, return empty string
        if len(string) == 0:
            return string
        if string[0] == ".":
            string = "0" + string

        # Splitting-based normalization

        # Process complex expressions without parentheses
        s_is_set = is_set(string)
        if s_is_set:
            raw_strings = self.extract_set(string)
        else:
            raw_strings = [string]

        strings = []
        for string in raw_strings:
            string = fix_sqrt(string)

            if string.startswith("frac"):
                string = "\\" + string
            # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
            string = fix_fracs(string)

            # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
            string = fix_a_slash_b(string)

            string = re.sub(r"^[a-z]\\in", "", string)

            if "," not in string:
                string = self.remove_out_paren(string)

            if "\\begin{array}" not in string:
                # to consider: get rid of chain of equalities like "a = b = c = d"
                if len(string.split("=")) > 2:
                    string = string.split("=")[-1]

                # to consider: get rid of e.g. "k = " or "q = " at beginning
                if len(string.split("=")) == 2:
                    first_part = string.split("=")[0].strip()
                    if (
                        re.match(
                            r"^([a-z]|[A-Z]{2}|\\?(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|sin|cos|sec|csc|tan|cot|sinh|cosh|sech|csch|tanh|coth|log|ln|exp))\^?{?-?('|\\prime|\d)*}?(\(-?([\d\.]+|[a-z])?\))?$",
                            first_part,
                        )
                        is not None
                    ):
                        string = string.split("=")[1]

                # to consider: get rid of equalities but not equations
                if len(string.split("=")) == 2:
                    if len(re.findall(r"[a-zA-Z]", string.split("=")[0].strip())) == 0:
                        string = string.split("=")[1]
            # replace \pm with +,-
            # string = re.sub(r"(.*?)\\pm(.+?)", r"\1-\2,\1+\2", string)
            string = self.norm_pm(string)  # might add comma ","

            string = re.sub(r"^0+([1-9])", r"\1", string)

            strings.append(string)
        string = ",".join(strings)

        if "," not in string:
            string = self.remove_out_paren(string)

        if STR2NUM.get(string):
            string = str(STR2NUM[string])

        # add space
        string = re.sub(r"\\mid([a-z])", r"\\mid \1", string)
        string = self.clean(string)

        # If there are multiple same inequality signs and no commas
        for ineq in ["<", ">"]:
            if len(re.findall(f"{ineq}=?", string)) > 1 and not any(
                delim in string.lower() for delim in [",", "and", "or"]
            ):
                string = string.replace(ineq, ",")

        # deal with abs
        if "\\abs(" in string:
            string = string.replace("\\abs(", "Abs(")
            string = parse_expr(string).evalf()

        return string
    
    # 在进行数值计算前，需要将sympy中的pi符号替换为pi的近似数值
    def sympy_sub_pi(self, expression_sympy):
        return expression_sympy.subs(self.pi, math.pi)
    
    def clean(self, ans: str) -> str:
        """Clean the extracted answer."""

        ans = ans.strip()
        ans = self.clean_preceding(ans)
        ans = self.clean_trailing(ans)

        return ans

    def normalize_abs_latex(self, expr):
        """
        Converts LaTeX absolute value expressions from \abs{...} or \abs(...) to \Abs{...}
        """
        # Replace \abs{...} with \Abs{...}
        expr = re.sub(r'\\abs\s*{([^}]*)}', r'\\Abs{\1}', expr)
    
        # Replace \abs(...) with \Abs{...}
        expr = re.sub(r'\\abs\s*\(([^)]*)\)', r'\\Abs{\1}', expr)
    
        return expr

    def clean_preceding(
        self,
        s: str,  # The input string.
    ) -> str:  # The cleaned string with preceding punctuation marks removed.
        """Removes preceding punctuation marks from a string."""
        s = str(s).strip()
        while s != "" and s[0] in NO_PRECEDING_PUNCS:
            s = s[1:].strip()

        return s

    def clean_trailing(
        self,
        s: str,  # The input string.
    ) -> str:  # The cleaned string with trailing punctuation marks removed.
        """Removes trailing punctuation marks from a string."""
        s = str(s).strip()
        while s != "" and s[-1] in NO_TRAILING_STRS:
            s = s[:-1].strip()
        return s
    
    def extract_boxed_answer(self, text):
        # extract answer wrapped in \boxed{} from models' output
        # TODO: add other extraction pattern
        # last boxed only
        content = remove_boxed(last_boxed_only_string(text))
        if content == None:
            match = re.search(r'\\boxed{', text)
            if match:
                start_index = match.end()
                end_index = start_index
                stack = 1
                while stack > 0 and end_index < len(text):
                    if text[end_index] == '{':
                        stack += 1
                    elif text[end_index] == '}':
                        stack -= 1
                    end_index += 1
                if stack == 0:
                    content = text[start_index:end_index - 1]
                    if not content:
                        return text
                    else:
                        content = self.normalize_answer(content)
                        return content
        if content == None:
            return text
        content = self.normalize_answer(content)
        return content
    
    def extract_ans(self, resp_str: str) -> str:
        """Extract answer segment from complete `resp`."""
        ans = self.extract_explicit_ans(resp_str)
        if ans is not None:
            return ans
        elif not self.strict_extract:
            # Speculate with the last latex formula
            matches = re.findall(
                r"(?:\$|\\\(|\\\[)([^\$]+)(?:\$|\\\(|\\\[)", resp_str, re.DOTALL
            )
            if len(matches) > 0:
                return matches[-1]
            # Speculate with the last number
            matches = re.findall(r"-?\d*\.?\d+", resp_str.replace(",", ""))
            if len(matches) > 0:
                return matches[-1]
        return ""  # Empty str if no answer is found


    def extract_explicit_ans(self, resp_str: str) -> str:
        resp_str = self.clean_trailing(resp_str)
        # might be answer only
        if "herefore" in resp_str:
            resp_str = resp_str.split("herefore")[-1].strip()
        if GSM8K_ANS_PREFIX in resp_str:
            resp_str = resp_str.split(GSM8K_ANS_PREFIX)[-1].strip()
        if PRM800K_ANS_PRRFIX in resp_str:
            resp_str = resp_str.split(PRM800K_ANS_PRRFIX)[-1].strip()

        if "oxed{" in resp_str:
            resp = self.extract_boxed_answer(resp_str)
        else:
            resp = resp_str

            # should be answer only
            if "is the ans" in resp:
                resp = re.split(r"(,|\.|\!\|?)", resp.split("is the ans")[-2].strip())[
                    -1
                ].strip()
            elif "is our ans" in resp:
                resp = re.split(r"(,|\.|\!\|?)", resp.split("is our ans")[-2].strip())[
                    -1
                ].strip()
            elif "answer is" in resp:
                resp = resp.split("answer is")[-1].strip()
            elif "answer:" in resp:
                resp = resp.split("answer:")[-1].strip()
            elif "answer :" in resp:
                resp = resp.split("answer :")[-1].strip()
            #elif "statement" in resp:
            #    bool_resp = norm_str2bool(resp.split("is ")[-1].strip())
            #    if bool_resp is not None:
            #        return str(bool_resp)
            else:
                return None

            if resp.startswith("$") and resp.endswith("$"):
                resp = resp[1:-1]

        return resp
        
     
    def split_by_comma(self, expr: str):
        # Splits expressions by commas outside of brackets
        # 用于处理逗号的嵌套情况
        # 例子: "f(x, y, z), g(a, b, c), h(i, j)"
        # deal with set
        expr = expr.replace("\\{", "(")
        expr = expr.replace("\\}", ")")
        expr = expr.replace("\\rangle", ")")
        expr = expr.replace("\\langle", "(")
        
        in_bracket_num = 0 # 这个值为0时，说明当前不在括号内部
        splitted_expr = []
        start_idx = 0
        for i, char in enumerate(expr):
            if char in ["(", "[", "<"]:
                in_bracket_num += 1
            elif char in [")", "]", ">"]:
                in_bracket_num -= 1
            elif char == "," and in_bracket_num == 0:
                splitted_expr.append(expr[start_idx:i].strip())
                start_idx = i + 1

        if start_idx < len(expr):
            splitted_expr.append(expr[start_idx:].strip())  
            
        if splitted_expr:
            splitted_expr = [item.strip("$").strip() for item in splitted_expr] 
        
        return splitted_expr
    
    def get_domain_of_definition(self, expr, symbol):
        """Determine domain of definition for a given symbol."""
        domain = continuous_domain(expr, symbol, S.Reals)
        intervals = []

        if domain.is_Union:
            # Handle multiple connected components
            for interval in domain.args:
                start = float(interval.start) if interval.start is not None else None
                end = float(interval.end) if interval.end is not None else None
                intervals.append((start, end))
        elif domain.is_Interval:
            start = float(domain.start) if domain.start is not None else None
            end = float(domain.end) if domain.end is not None else None
            intervals.append((start, end))
        
        # Return a list of intervals
        return intervals if intervals else [(-10, 10)]


    def random_value_check(self, expr):
        """Check if the expression evaluates close to zero for random values considering domain."""
        symbols_in_expr = expr.free_symbols  # Extract all symbols in the expression
        for _ in range(self.num_times):
            subs_dict = {}
            for symbol in symbols_in_expr:
                intervals = self.get_domain_of_definition(expr, symbol)
                if not intervals:
                    continue  # Skip if no valid intervals, although fallback provided

                # Choose a random interval and then a value within that interval
                selected_interval = random.choice(intervals)
                start, end = selected_interval
                if start == -np.inf or end == np.inf:
                    # Handle infinite bounds, use large finite bounds for practical sampling
                    start = start if start != -np.inf else -1e4
                    end = end if end != np.inf else 1e4
                
                subs_dict[symbol] = (start, end)#np.random.uniform(start, end, self.num_samples)
            
            for __ in range(self.num_samples):
                for k, v in subs_dict.items():
                    ddd = {}
                    ddd[k] = random.uniform(v[0], v[1])
                evaluated_expr = expr.evalf(subs=ddd)
                if abs(evaluated_expr) > self.precision:
                    return False
        return True

    def auto_judge(self, pred, gold, options, type_sequence=None, precision=1e-8):
        
        def handler(signum, frame):
            raise Exception("Time out!")
            
        signal.signal(signal.SIGALRM, handler)

        # TODO: adjust extract answer patterns accordingly
        extracted_pred = self.extract_ans(pred)
        if not extracted_pred: # no answer can be extracted in model's output
            return False
        
        # deal with predition list
        extracted_pred = self.split_by_comma(extracted_pred)
        #extracted_pred = [self.normalize_answer(item) for item in extracted_pred]
        if type_sequence != None:
            judge_tf_list = [item == "TF" for item in type_sequence]
        extracted_pred = [self.norm_ans_str(item) for item in extracted_pred]
        gold = [self.norm_ans_str(item) for item in gold]
        if type_sequence != None:
            extracted_pred = [self.norm_ans_str(item, tf) for item, tf in zip(extracted_pred, judge_tf_list)]
            gold = [self.norm_ans_str(item, tf) for item, tf in zip(gold, judge_tf_list)]
            
            


        # if number of predicted answers != number of ground-truth
        if len(extracted_pred) != len(gold):
            return False

        # deal with precision list
        precision = precision if type(precision) == list else [precision]
        precision = precision * len(gold)
        for item1, item2, pre, opt in zip(extracted_pred, gold, precision, options):
            self.precision = pre
            if not self.is_equal(item1, item2, options=opt):
                return False
        return True
    
    def judge(self, pred, gold, type_sequence, options, precision=1e-8):
        """
        Args:
            pred (str): the model's complete response
            gold (str): the ground truth answer
            type_sequence (list of str, optional): if the problem contains multiple answers, the list contains each answer's type. Defaults to None.

        Returns:
            bool: True/False
        """
        #assert len(gold) == len(type_sequence) == len(options)
        #if len(gold) != len(type_sequence) or len(gold) != len(options) or len(options) != len(type_sequence):
        #    print(gold)
        
        extracted_pred = self.extract_ans(pred)
        if not extracted_pred: # no boxed answer in model's output
            return False

        # deal with predition list
        extracted_pred = self.split_by_comma(extracted_pred)
        #extracted_pred = [self.normalize_answer(item) for item in extracted_pred]
        #judge = [True if item in ["MCS", "MCM"] else False for item in type_sequence]
        #ddd = [item if item is not None else None for item in extracted_pred]
        #extracted_pred = [self.norm_ans_str(item, j) for item, j in zip(ddd, judge)]

        #gold = [self.norm_ans_str(item, j) for item, j in zip(gold, judge)]
        #extracted_pred = [self.norm_ans_str(item) for item in extracted_pred]
        #gold = [self.norm_ans_str(item) for item in gold]

        #judge_tf_list = [item == "TF" for item in type_sequence] 
        extracted_pred = [self.norm_ans_str(item, tf) for item, tf in zip(extracted_pred, type_sequence)]
        gold = [self.norm_ans_str(item, tf) for item, tf in zip(gold, type_sequence)]

        # if number of predicted answers != number of ground-truth
        if len(extracted_pred) != len(gold):
            return False

        # deal with precision list 
        precision = precision if type(precision) == list else [precision]
        precision = precision * len(gold) 

        for item1, item2, pre, answer_type, opt in zip(extracted_pred, gold, precision, type_sequence, options):
            self.precision = pre
            try:
                if not self.judgment_methods[answer_type](item1, item2, options=opt):
                    return False
            except:
                return False
        return True 

    def aux_judge(self, pred, gold, question):
        initialize_client()
        with open('./data/judge_prompt.txt', 'r') as file:
            judge_prompt = file.read()
        #if gold == None:
        #    return False, None
        if pred == None:
            return False, None
        
        judge_prompt = judge_prompt.replace("{{problem}}", question).replace("{{Reference Answer}}", ", ".join(gold)).replace("{{Solution}}", pred)
        success = False
        while not success:
            try:
                response = client.chat.completions.create(
                    model=self.judge_model,
                    messages=[
                        {
                            "role": "user",
                            "content": judge_prompt,
                        },
                    ],
                    max_tokens=1024,
                    temperature=0.0,
                    n=1
                ) 
                res = response.choices[0].message.content
            except Exception as e:
                time.sleep(random.randint(1,30))
                print(f"{e}")
            except openai.error.APIerror:
                time.sleep(random.randint(1,30))
                print(f"APIError了。")
            else:
                success = True
                break
        try:
            correctness = res.split("## Justification")[0].split("## Equivalence Judgement")[-1].strip()
        except Exception as e:
            print(e)
        if correctness == "TRUE":
            return True, res
        else:
            return False, res

    def is_equal(self, ans, gold, options = [], exclude = None):
        answer_type_list = self.judgment_methods.keys()
        if exclude != None:
            answer_type_list = [item for item in answer_type_list if item not in exclude]
        for answer_type in answer_type_list:
            try:
                #signal.alarm(5)
                if self.judgment_methods[answer_type](ans, gold):
                #    signal.alarm(0)
                    return True
                #signal.alarm(0)
            except:
                pass
            finally:
                signal.alarm(0)
        return False


    def judge_single_numerical_value(self, pred, gold, options=[]):
        def is_scientific_notation(expr):
            return isinstance(expr, Mul) and isinstance(expr.args[1], Pow) and expr.args[1].args[0] == 10

        def to_scientific_notation_latex(num):
            num_sci = f"{num:.2e}"
            base, exponent = num_sci.split('e')
            exponent = int(exponent)
            return f"{base}\\times 10^{{{exponent}}}"
        
        # remove unit ?

        # pure value -> can be parsed by python
        if pred == gold: # exact the same
            return True
        try: # can be parsed by python directly
            pred_value = float(pred)
            gold_value = round(float(gold), 6)
            if abs((pred_value - gold_value)/gold_value) <= self.precision * 1.01:
                return True
        except:
            pass
        # cannot be parsed by python, use scipy expression to judge
        # like 2^5, \log _2 7
        try:
        #breakpoint()
            exp_pred = self.sympy_sub_pi(sympify(parse_latex(pred)))
            #breakpoint()
            exp_gold = self.sympy_sub_pi(sympify(parse_latex(gold)))
            #if abs(N(exp_pred) - N(exp_gold)) <= self.precision * 1.01:
            #    return True
            if abs((exp_pred.evalf() - exp_gold.evalf())/exp_gold.evalf()) <= self.precision * 1.01:
                return True 
            if is_scientific_notation(exp_pred) != is_scientific_notation(exp_gold):
                if is_scientific_notation(exp_pred):
                    gold = to_scientific_notation_latex(float(gold))
                    exp_gold = parse_latex(gold)
                else:
                    pred = to_scientific_notation_latex(float(pred))
                    exp_pred = parse_latex(pred)
                
            if is_scientific_notation(exp_pred) and is_scientific_notation(exp_gold):
                base_pred, exponent_pred = N(exp_pred.args[0]), N(exp_pred.args[1].args[1])
                base_gold, exponent_gold = N(exp_gold.args[0]), N(exp_gold.args[1].args[1])
                if exponent_pred == exponent_gold and abs(base_pred-base_gold) <= 0.1*1.01:
                    return True
            else:
                if N(exp_pred) == N(exp_gold):
                    return True
        except:
            pass
        
        return False


    def judge_MC_single(self, pred, gold, options=[]):
        # TODO: add MC with options that are not ABCD
        if options == []:
            common_answer = [chr(i) for i in range(65, 91)] # 'A'~'Z'
        else:
            #common_answer = options
            common_answer = [item.lower() for item in options]
            pred = pred.lower()
            gold = gold.lower()
        if pred.lower() == gold.lower():
            return True
        else:
            if pred.startswith("[") and pred.endswith("]"):
                pred = pred.strip("[]")
            if pred[0] in common_answer and (len(pred) > 1 and pred[1] == ":"):
                return pred[0] == gold
            else:
                return False

    def judge_MC_multiple(self, pred, gold, options=[]):
        # TODO: add MC with options that are not ABCD 
        if options == []:
            common_answer = [chr(i) for i in range(65, 91)] # 'A'~'Z'
        else:
            common_answer = [item.lower() for item in options]
            pred = pred.lower()
            gold = gold.lower()

        gold_list = [item for item in gold]
        pred_list = [item for item in pred if item in common_answer]
        if len(gold_list) != len(pred_list):
            return False
        
        # ignore order
        idx = -1
        while len(gold_list) != 0:
            idx = (idx + 1) % len(gold_list)

            item1 = gold_list[idx]

            for item2 in pred_list:
                if item1.lower() == item2.lower():
                    gold_list.remove(item1)
                    pred_list.remove(item2)
                    break
            else:
                # If we didn't break from the inner loop, it means no match was found
                return False

        # If all elements are matched and removed, the lists can be paired
        return True
    
    def judge_equation(self, pred, gold, **kwargs):
        def simplify_equation(latex_eq):
            try:
                lhs, rhs = latex_eq.split('=')
            except:
                lhs = latex_eq
                rhs = "0"
            
            lhs_expr = parse_latex(lhs)
            rhs_expr = parse_latex(rhs)
            equation = Eq(lhs_expr, rhs_expr)
            try:
                simplified_eq = simplify(equation.lhs - equation.rhs)
            except:
                simplified_eq = simplify(lhs_expr - rhs_expr)
            return simplified_eq
        try:
            expr1_sym = simplify_equation(pred)
            expr2_sym = simplify_equation(gold)
            difference = simplify(expr1_sym - expr2_sym)
            
            if difference == 0:
                return True
            else:
                division_result_1 = simplify(expr1_sym / expr2_sym)
                division_result_2 = simplify(expr2_sym / expr1_sym)
                if (division_result_1.is_Integer and division_result_1 != 0) or (division_result_2.is_Integer and division_result_2 != 0):
                    return True
                else:
                    return False
        except Exception as e:
            print(e)
            return False
    
    def judge_expression(self, pred, gold, **kwargs):
        def extract_expression(expression):
            if "=" in expression:
                expression = expression.split("=")[1]
            return expression.strip()
        exp1 = extract_expression(pred)
        exp2 = extract_expression(gold)
        expr1_sym = trigsimp(self.sympy_sub_pi(sympify(parse_latex(exp1))))
        expr2_sym = trigsimp(self.sympy_sub_pi(sympify(parse_latex(exp2))))
        
        if expr1_sym == expr2_sym:
            return True
        else:
            # judge if the expression contains symbol(like x, y)
            #if (expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol)) or (not expr1_sym.has(sp.Symbol) and expr2_sym.has(sp.Symbol)):
            #    return False
            if not expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol):
                try:
                    return self.judge_single_numerical_value(expr1_sym, expr2_sym)
                except:
                    return False
            else:
                try:
                    simplified_expr = simplify(expr1_sym - expr2_sym)
                    num_value = simplified_expr.evalf()
                    if num_value.has(sp.Symbol):
                        return self.random_value_check(simplified_expr)
                    flag = bool(abs(num_value) < self.precision)
                    assert type(flag) == bool
                    return flag
                except:
                    return False
    
    def judge_extract_match(self, pred, gold, **kwargs):
        return pred.lower() == gold.lower()

    def judge_TF(self, pred, gold, **kwargs):
        answers = ["True", "False"]
        assert gold in answers
        if pred not in answers:
            return False
        else:
            return pred == gold
    
    def judge_interval(self, pred, gold, **kwargs):
        def parse_interval(interval):
            # Parse the interval string and return a list of tuples. Each tuple contains the interval values and types.
            parsed = []
            for part in interval.split('\\cup'):
                bounds, interval_type = part.strip(), ''
                if bounds.startswith('('):
                    interval_type += 'open_left'
                else:
                    interval_type += 'closed_left'
                if bounds.endswith(')'):
                    interval_type += '_open_right'
                else:
                    interval_type += '_closed_right'
                # Remove the interval characters to just get the numbers
                numbers = bounds.strip('()[]').split(',')
                parsed.append((numbers, interval_type))
            return parsed
        
        def compare_intervals(intervals1, intervals2):
            list1 = [(tuple(item[0]), item[1]) for item in intervals1]
            list2 = [(tuple(item[0]), item[1]) for item in intervals2]

            if len(list1) != len(list2):
                return False

            # Compare each parsed interval from list1 against all in list2
            for interval1 in list1:
                interval_numbers1, interval_type1 = interval1
                matched = False
                for interval2 in list2:
                    interval_numbers2, interval_type2 = interval2
                    # First check if the types of intervals match
                    if interval_type1 == interval_type2:
                        # Then check if both bounds of the intervals are mathematically equal
                        bounds_match = self.judge_expression(interval_numbers1[0], interval_numbers2[0]) and \
                                        self.judge_expression(interval_numbers1[1], interval_numbers2[1])
                        if bounds_match:
                            matched = True
                            list2.remove(interval2)
                            break
                if not matched:
                    return False
            return True
        
        # Parse both interval expressions
        parsed_intervals1 = parse_interval(pred)
        parsed_intervals2 = parse_interval(gold)

        # Compare the parsed intervals
        return compare_intervals(parsed_intervals1, parsed_intervals2)
    
    def judge_ordered_list(self, pred, gold, **kwargs):
        pred_list = self.split_by_comma(pred.strip("()<>"))
        gold_list = self.split_by_comma(gold.strip("()"))
        if len(pred_list) != len(gold_list):
            return False
        else:
            for i in range(len(pred_list)):
                if not self.is_equal(pred_list[i], gold_list[i], exclude=EXCLUDE_TYPE):
                    return False
            return True
        
    def judge_unordered_list(self, pred, gold, **kwargs):
        pred_list = self.split_by_comma(pred.strip("()<>"))
        gold_list = self.split_by_comma(gold.strip("()"))
        if len(pred_list) != len(gold_list):
            return False
        idx = -1
        while len(gold_list) != 0:
            idx = (idx + 1) % len(gold_list)

            item1 = gold_list[idx]

            for item2 in pred_list:
                if self.is_equal(item2, item1, exclude=EXCLUDE_TYPE):
                    gold_list.remove(item1)
                    pred_list.remove(item2)
                    break
            else:
                # If we didn't break from the inner loop, it means no match was found
                return False

        # If all elements are matched and removed, the lists can be paired
        return True


if __name__ == "__main__":
    judger = Judger()
    #pred = "\\boxed{[A,B]}"
    #gold = "A,B"

    # Trigonometry

    # test numerical value

    #gold = ['1.01']
    #pred = "\\boxed{1.01}"

    #gold = ["\\sin(43.5558*\\pi/180)"]
    #pred = "\\boxed{0.6890606866870983}"

    #gold = ["\\sqrt{3}"]
    #pred = "\\boxed{1.7320508075688772}"

    # TODO: asin -> arcsin; abs
    #gold = ["\\sqrt{3}", "\\sin(43.5558*\\pi/180)", "arcsin(1)", "arcsin{1}"]
    #pred = "\\boxed{1.7320508075688772,0.6890606866870983, \\pi/2, \\pi/2}"
    #pred = "\\boxed{0.6890606866870983, 1.7320508075688772}"
    #ts = ['NV', "NV", "NV", "NV"]

    #print(judger.judge(pred, gold, ts))
    #print(judger.auto_judge(pred, gold))

    #gold = ["sin(43.5558*\\pi/180)"]
    #pred = "\\boxed{0.6890606866870983}"
    #ts = ["NV"]

    #gold = ["sin(43.5558*pi/180)"]
    #pred = "\\boxed{0.6890606866870983}"
    #ts = ["NV"]


    # test 3e+5
    #gold = ["1.5e+3"]
    #pred = "\\boxed{1500}"
    #ts = ["NV"]
    #aaa = [[]]
    #print(judger.judge(pred, gold, ts, aaa)) 

    #gold = ["1.5E+3"]
    #pred = "\\boxed{1500}"
    #ts = ["NV"]
    #aaa = [[]]
    #print(judger.judge(pred, gold, ts, aaa)) 

    #gold = ["\\abs(-1)"]
    #pred = "\\boxed{1}"
    #ts = ["NV"]
    #print(judger.judge(pred, gold, ts, [[]]))
    #print(judger.auto_judge(pred, gold))

    # test MC with sinlge choice

    #gold = ['A', "C", "B"]
    #pred = "\\boxed{A, C, B}"
    #ts = ['MCS', 'MCS', 'MCS']
    #print(judger.judge(pred, gold, ts))
    #print(judger.auto_judge(pred, gold))


    # test MC with multiple choices
    #gold = ["ACE", "BC"]
    #pred = "\\boxed{ACE, BC}"
    #ts = ["MCM", "MCM"]

    #gold = ["AEC", "CB"]
    #pred = "\\boxed{ACE, BC}"
    #ts = ["MCM", "MCM"]
    
    #print(judger.judge(pred, gold, ts))
    #print(judger.auto_judge(pred, gold))

    # test HI
    #gold = ["T", "F", "converge"]
    #pred = "\\boxed{T, F, converge}"
    #ts = ["HI", "HI", "HI"]

    #gold = ["[cos(C)]^2-[sin(C)]^2"]
    #pred = "\\boxed{cos(C)^2 - sin(C)^2}"
    #ts = ["EX"]

    #gold = ["2*6x"]
    #pred = "\\boxed{12x}"
    #ts = ["EX"]
    #aaa = [[]]
    #print(judger.judge(pred, gold, ts, options=aaa))
    #print(judger.auto_judge(pred, gold))


    # test equation
    #gold = ["y = 8"]
    #pred = "\\boxed{y = 8}"
    #ts = ['EQ']

    #print(judger.judge(pred, gold, ts, [[]]))
    #print(judger.auto_judge(pred, gold)) 

    # test expression
    #gold = ["\\cos(\\pi/2*(x+1))+2"]
    #pred = "\\boxed{1 + 1 - \\sin(\\pi/2*x)}"
    #ts = ['EX']
    
    #print(judger.judge(pred, gold, ts))
    #print(judger.auto_judge(pred, gold))  

    # test TF
    #gold = ["T", "F"]
    #pred = "\\boxed{Yes, FALSE}"
    #ts = ["TF", "TF"]
    #print(judger.judge(pred, gold, ts)) 


    # test OL
    gold = ["(3.1415926535898, F)"]
    pred = "\\boxed{(pi, No)}"
    ts = ["OL"]
    aaa = [[]]
    print(judger.judge(pred, gold, ts, aaa)) 

    # test set as UOL
    #gold = ["{3.1415926535898, F}"]
    #pred = "\\boxed{(pi, No)}"
    #ts = ["UOL"]
    #aaa = [[]]
    #print(judger.judge(pred, gold, ts, aaa))  

    # test special OL
    #gold = ["(e, x1)"]
    #pred = "\\boxed{(e, x)}"
    #ts = ["OL"]
    #aaa = [[]]
    #print(judger.judge(pred, gold, ts, aaa))  

    # test UOL
    #gold = ["3.1415926535898, F"]
    #pred = "\\boxed{(No, pi)}"
    #ts = ["UOL"]
    #print(judger.judge(pred, gold, ts)) 


    # extract answer
    #gold = ["0",
    #        "121",
    #        "20",
    #        "5",
    #        "93",
    #        "28",
    #        "0.24",
    #        "0.04"]
    #pred = "(a) $t^5$: Since $t^2 = 11$, we have $t^5 = t^3 \\cdot t^2 = t^3 \\cdot 11 = (t^2 \\cdot t) \\cdot 11 = (11 \\cdot t) \\cdot 11 = 121t$. Therefore, $t^5 = \\boxed{0} + \\boxed{121}t$.\n\n(b) $(6-t)(7+2t)$: Expanding, we get $(6-t)(7+2t) = 42 + 12t - 7t - 2t^2 = 42 + 5t - 2 \\cdot 11 = 42 + 5t - 22 = 20 + 5t$. Therefore, $(6-t)(7+2t) = \\boxed{20} + \\boxed{5}t$.\n\n(c) $(7+2t)^2$: Expanding, we get $(7+2t)^2 = 49 + 28t + 4t^2 = 49 + 28t + 4 \\cdot 11 = 49 + 28t + 44 = 93 + 28t$. Therefore, $(7+2t)^2 = \\boxed{93} + \\boxed{28}t$.\n\n(d) $1/(6-t)$: Since $t^2 = 11$, we have $1/(6-t) = 1/(6-t) \\cdot (6+t)/(6+t) = (6+t)/((6-t)(6+t)) = (6+t)/(36-t^2) = (6+t)/(36-11) = (6+t)/25 = 6/25 + t/25$. Therefore, $1/(6-t) = \\boxed{\\frac{6}{25}} + \\boxed{\\frac{1}{25}}t$.\n\nThe final answers are $\\boxed{0, 121, 20, 5, 93, 28, \\frac{6}{25}, \\frac{1}{25}}$."
    #ts = ["NV", "NV", "NV", "NV", "NV", "NV", "NV", "NV"]
    #aaa = [[], [], [], [], [], [], [], []]
    #print(judger.judge(pred, gold, ts, aaa))  

    # test multivariable
    #gold = ["(-1, 2, 3) + t(2, -2, -2)"]
    #pred = "\\boxed{2t(1, -1, -1) + (-1, 2, 3)}"
    #ts = ['EX']
    #aaa = [[]]
    #print(judger.judge(pred, gold, ts, aaa))  

    # test geometry wrong case
    #gold = ["(-2, 11, -5)"]
    #pred = "\\boxed{\\left<-2, 11, -5\\right>}."
    #ts = ["OL"]
    #aaa = [[]]
    #pred = "\\boxed{6\\,\\mathit{\\vec j} + 7\\,\\mathit{\\vec k}}"
    #gold = ["6j+7k"]
    #ts = ["OL"]
    #aaa = [[]]
    #print(judger.judge(pred, gold, ts, aaa))  

    # test Statistics wrong case
    #gold = ["PARAMETER",
    #        "PARAMETER",
    #        "STATISTIC",
    #        "statistic"]
    #pred = "\\boxed{parameter, parameter, statistic, statistic}"
    #ts = ["MCS", "MCS", "MCS", "MCS"]
    #aaa = [["PARAMETER",
    #        "STATISTIC"],
    #        ["PARAMETER",
    #        "STATISTIC"],
    #        ["PARAMETER",
    #        "STATISTIC"],
    #        ["PARAMETER",
    #        "STATISTIC"]]
    #print(judger.judge(pred, gold, ts, aaa))


    # test set as UOL
    #pred = "\\boxed{\\{2, 4, 8\\}, \\{1, 2, 3, 4, 7, 8\\}}"
    #gold = ["(2, 4, 8)", "(1, 2, 3, 4, 7, 8)"]
    #ts = ["UOL", "UOL"]
    #aaa = [[], []]
    #print(judger.judge(pred, gold, ts, aaa))
