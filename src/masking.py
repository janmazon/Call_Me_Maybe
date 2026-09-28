from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition
from typing import Any
import json
import re


def get_allowed_tokens(candidates: list[list[int]],
                       generated_tokens: list[int]) -> list[int]:
    allowed_token_ids: list[int] = []
    position = len(generated_tokens)
    for candidate in candidates:
        if (len(candidate) > position and
                candidate[:position] == generated_tokens):
            if candidate[position] not in allowed_token_ids:
                allowed_token_ids.append(candidate[position])
    return allowed_token_ids


def apply_logit_mask(logits: list[Any], allowed_token_ids: list[int]) -> int:
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


def load_vocabulary(model: Small_LLM_Model) -> dict[int, str]:
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r") as f:
        vocab = json.load(f)

        inverted_vocab: dict[int, str] = {}
        for string_token, num_id in vocab.items():
            inverted_vocab[num_id] = string_token

    return inverted_vocab


def check_json_rules(function: FunctionDefinition, proposed_text: str) -> bool:
    remaining_text = proposed_text
    is_first = True

    for param_name, param_def in function.parameters.items():
        if is_first:
            expected_text = f'{{"{param_name}": '
            is_first = False
        else:
            expected_text = f', "{param_name}": '

        if len(remaining_text) <= len(expected_text):
            if expected_text.startswith(remaining_text):
                return True
        else:
            if remaining_text.startswith(expected_text):
                remaining_text = remaining_text[len(expected_text):]
            else:
                return False

        if param_def.type == "number":
            number_regex = r"-?[0-9]+(\.[0-9]+)?"
            match = re.match(number_regex, remaining_text)
            if match and len(match.group(0)) < len(remaining_text):
                remaining_text = remaining_text[len(match.group(0)):]
            else:
                if re.fullmatch(r"^-?[0-9]*\.?[0-9]*$", remaining_text):
                    return True
                else:
                    return False

        elif param_def.type == "string":
            string_regex = r'"[^"]*"'
            match = re.match(string_regex, remaining_text)
            if match and len(match.group(0)) < len(remaining_text):
                remaining_text = remaining_text[len(match.group(0)):]
            else:
                if re.fullmatch(r'^"[^"]*$', remaining_text):
                    return True
                else:
                    return False

    if remaining_text in ["", "}"]:
        return True

    return False


def get_allowed_tokens_for_args(function: FunctionDefinition,
                                generated_tokens: list[int],
                                inverted_vocab: dict[int, str]) -> list[int]:
    generated_text = ""
    for num in generated_tokens:
        generated_text += inverted_vocab[num]

    allowed: list[int] = []
    for num_id, string_token in inverted_vocab.items():
        proposed_text = generated_text + string_token
        if check_json_rules(function, proposed_text):
            allowed.append(num_id)

    return allowed
