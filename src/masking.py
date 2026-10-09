from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition
from typing import Any
import re


def get_allowed_tokens(candidates: list[list[int]],
                       generated_tokens: list[int]) -> list[int]:
    """
    Finds out which tokens are allowed next when picking a function.

    It compares the tokens made so far with the list of possible valid tokens.

    Args:
        candidates (list[list[int]]): List of valid token sequences.
        generated_tokens (list[int]): Tokens made so far.

    Returns:
        list[int]: A list of token IDs allowed for the next step.
    """

    allowed_token_ids: list[int] = []
    position = len(generated_tokens)
    for candidate in candidates:
        if (len(candidate) > position and
                candidate[:position] == generated_tokens):
            if candidate[position] not in allowed_token_ids:
                allowed_token_ids.append(candidate[position])
    return allowed_token_ids


def apply_logit_mask(logits: list[Any], allowed_token_ids: list[int]) -> int:
    """
    Applies a math mask to the model's probabilities.

    It finds and returns the token ID with the highest score, but only looks
    at tokens allowed by the rules.

    Args:
        logits (list[Any]): Raw probabilities from the model.
        allowed_token_ids (list[int]): IDs of tokens allowed by the rules.

    Returns:
        int: The winning token ID for the model to generate.
    """

    if logits and isinstance(logits[0], list):
        last_logits = logits[-1]
    else:
        last_logits = logits

    best_token = allowed_token_ids[0]
    best_score = last_logits[best_token]

    for token_id in allowed_token_ids:
        score = last_logits[token_id]
        if score > best_score:
            best_score = score
            best_token = token_id

    return best_token


def select_function(
        model: Small_LLM_Model,
        functions: list[FunctionDefinition],
        input_ids: list[int],
        ) -> tuple[FunctionDefinition, list[int]]:
    """
    Forces the model to output the exact name of an available function.

    It uses constrained decoding to stop the model from making up names.

    Args:
        model (Small_LLM_Model): The language model object.
        functions (list[FunctionDefinition]): The candidate functions.
        input_ids (list[int]): IDs of the first tokens from the prompt.

    Returns:
        tuple[FunctionDefinition, list[int]]: The chosen function and the
        updated list of tokens with the name added.
    """

    candidates: list[list[int]] = []
    function_by_tokens: dict[tuple[int, ...], FunctionDefinition] = {}
    current_input_ids = list(input_ids)
    generated_tokens: list[int] = []

    for function in functions:
        raw_tokens = model.encode(function.name).tolist()
        if raw_tokens and isinstance(raw_tokens[0], list):
            tokens = raw_tokens[0]
        else:
            tokens = raw_tokens
        candidates.append(tokens)
        function_by_tokens[tuple(tokens)] = function

    while True:
        allowed = get_allowed_tokens(candidates, generated_tokens)
        if not allowed:
            break
        logits = model.get_logits_from_input_ids(current_input_ids)
        next_token = apply_logit_mask(logits, allowed)
        generated_tokens.append(next_token)
        current_input_ids.append(next_token)

    selected_function = function_by_tokens[tuple(generated_tokens)]

    return selected_function, current_input_ids


def check_json_rules(function: FunctionDefinition, proposed_text: str) -> bool:
    """
    Checks if the text being built follows JSON rules.

    It uses text patterns (regex) for numbers, booleans, and strings to make
    sure keys, quotes, and values are 100% correct.

    Args:
        function (FunctionDefinition): The function with the needed rules.
        proposed_text (str): The string built so far.

    Returns:
        bool: True if the text follows the rules, False if it breaks them.
    """

    remaining_text = proposed_text.lstrip()
    if not remaining_text:
        return True
    if "{".startswith(remaining_text) or remaining_text.startswith("{"):
        remaining_text = remaining_text[1:].lstrip()
    else:
        return False
    if not remaining_text:
        return True

    params = list(function.parameters.items())
    for index, (param_name, param_def) in enumerate(params):
        if index == 0:
            expected_key = f'"{param_name}":'
        else:
            expected_key = f',"{param_name}":'
        if expected_key.startswith(remaining_text):
            return True
        elif remaining_text.startswith(expected_key):
            remaining_text = remaining_text[len(expected_key):].lstrip()
        else:
            return False
        if not remaining_text:
            return True

        elif param_def.type == "number":
            match = re.match(r"[^,}]+", remaining_text)
            if match:
                value = match.group(0)
                if len(value) == len(remaining_text):
                    if re.fullmatch(r"^-?[0-9]*\.?[0-9]*([eE][+-]?[0-9]*)?$",
                                    value.strip()):
                        return True
                    else:
                        return False
                else:
                    full_match = (r"^-?[0-9]+(\.[0-9]+([eE][+-]?[0-9]+"
                                  r")?|[eE][+-]?[0-9]+)$")
                    if not re.fullmatch(full_match, value.strip()):
                        return False
                    remaining_text = remaining_text[len(value):].lstrip()
            else:
                return False

        elif param_def.type == "integer":
            match = re.match(r"[^,}]+", remaining_text)
            if match:
                value = match.group(0)
                if len(value) == len(remaining_text):
                    if re.fullmatch(r"^-?[0-9]*$", value.strip()):
                        return True
                    else:
                        return False
                else:
                    if not re.fullmatch(r"^-?[0-9]+$", value.strip()):
                        return False
                    remaining_text = remaining_text[len(value):].lstrip()
            else:
                return False

        elif param_def.type in ["bool", "boolean"]:
            match = re.match(r"[^,}]+", remaining_text)
            if match:
                value = match.group(0)
                fullmatch = r"^(t|tr|tru|true|f|fa|fal|fals|false)$"
                if len(value) == len(remaining_text):
                    if re.fullmatch(fullmatch, value.strip()):
                        return True
                    else:
                        return False
                else:
                    if not re.fullmatch(r"^(true|false)$", value.strip()):
                        return False
                    remaining_text = remaining_text[len(value):].lstrip()
            else:
                return False

        elif param_def.type == "string":
            match_complete = re.match(r'^"([^"\\]|\\.)*"', remaining_text)
            if match_complete:
                value = match_complete.group(0)
                remaining_text = remaining_text[len(value):].lstrip()
            else:
                if re.fullmatch(r'^"([^"\\]|\\.)*$', remaining_text):
                    return True
                else:
                    return False

    remaining_text = remaining_text.strip()
    if not remaining_text or remaining_text == "}":
        return True
    return False


def get_allowed_tokens_for_args(function: FunctionDefinition,
                                generated_tokens: list[int],
                                inverted_vocab: dict[int, str]) -> list[int]:
    """
    Gets all vocabulary tokens that keep the JSON valid.

    It tests all clean tokens by adding them to the current text and checking
    them against the rules.

    Args:
        function (FunctionDefinition): The rules to follow.
        generated_tokens (list[int]): Parameter tokens already made.
        inverted_vocab (dict[int, str]): Dictionary of tokens to text.

    Returns:
        list[int]: IDs of the tokens the model is allowed to generate.
    """

    generated_text = ""
    for num in generated_tokens:
        generated_text += inverted_vocab[num]

    allowed: list[int] = []
    for num_id, string_token in inverted_vocab.items():
        proposed_text = generated_text + string_token
        if check_json_rules(function, proposed_text):
            allowed.append(num_id)

    return allowed
