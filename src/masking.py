from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition
from typing import Any


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
