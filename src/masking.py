import numpy as np
from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition


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


def apply_logit_mask(logits: np.ndarray, allowed_token_ids: list[int]) -> int:
    logits_array = np.asarray(logits, dtype=np.float32).flatten()
    mask = np.full_like(logits_array, -np.inf)
    mask[allowed_token_ids] = logits_array[allowed_token_ids]
    return int(np.argmax(mask))


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
        tokens: list[int] = model.encode(function.name).tolist()
        candidates.append(tokens)
        function_by_tokens[tuple(tokens)] = function

    while True:
        allowed = get_allowed_tokens(candidates, generated_tokens)
        if not allowed:
            break
        logits = np.array(model.get_logits_from_input_ids(current_input_ids))
        next_token = apply_logit_mask(logits, allowed)
        generated_tokens.append(next_token)
        current_input_ids.append(next_token)

    selected_function = function_by_tokens[tuple(generated_tokens)]
    return selected_function, current_input_ids
