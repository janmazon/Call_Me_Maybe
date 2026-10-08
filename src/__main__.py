import json
import os
import time
import sys
import numpy as np
from typing import Any
from llm_sdk.llm_sdk import Small_LLM_Model
from src.parser import parse_arguments
from src.data_loader import load_functions, load_tests, load_vocabulary
from src.data_loader import create_clean_vocabulary
from src.generator import create_prompt
from src.masking import select_function, get_allowed_tokens_for_args
from src.models import FunctionCallResult, TestPrompt, FunctionDefinition


def process_test(test: TestPrompt, functions: list[FunctionDefinition],
                 model: Small_LLM_Model, clean_vocab: dict[int, str],
                 inverted_vocab: dict[int, str]) -> dict[str, Any]:
    print(f"Process: {test.prompt}")
    prompt_tokens = create_prompt(functions, test.prompt)
    input_ids = model.encode(prompt_tokens).tolist()[0]
    chosen_function, final_input_ids = select_function(model, functions,
                                                       input_ids)
    current_context_ids = list(final_input_ids)
    json_tokens_generated: list[int] = []
    max_tokens = 50

    for i in range(max_tokens):
        logits = model.get_logits_from_input_ids(current_context_ids)
        allowed_ids = get_allowed_tokens_for_args(chosen_function,
                                                  json_tokens_generated,
                                                  clean_vocab)
        if not allowed_ids:
            print("\n[Error: Validator blocked all tokens. Check rules.]")
            break
        mask = np.full_like(logits, -np.inf)
        mask[allowed_ids] = 0.0
        masked_logits = np.array(logits) + mask
        next_token_id = int(np.argmax(masked_logits))
        json_tokens_generated.append(next_token_id)
        current_context_ids.append(next_token_id)
        generated_text = ""
        for num in json_tokens_generated:
            generated_text += inverted_vocab[num]
        if (generated_text.endswith("}") and
                generated_text.count('"') % 2 == 0):
            break

    try:
        clean_text = generated_text.replace("Ġ", " ").replace("Ċ", "\n")
        json_data = json.loads(clean_text)
    except json.JSONDecodeError:
        print(f"Error parsinng generated JSON for prompt: {test.prompt}.")
        json_data = {}
    except KeyError as e:
        print(f"Missing key {e} in JSON for prompt: {test.prompt}.")
        json_data = {}

    result = FunctionCallResult(prompt=test.prompt,
                                name=chosen_function.name,
                                parameters=json_data)
    return result.model_dump()


def save_output(output_data: list[dict[str, Any]], output_path: str) -> None:
    dir = os.path.dirname(output_path)
    try:
        if dir:
            os.makedirs(dir, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
    except PermissionError:
        print("Error: No permissions to write to directory.")
        sys.exit(1)
    except OSError as e:
        print(f"Error: {e}")
        sys.exit(1)


def main() -> None:
    try:
        args = parse_arguments()
        functions = load_functions(args.functions_definition)
        tests = load_tests(args.input)
        model = Small_LLM_Model()
        inverted_vocab = load_vocabulary(model)
        clean_vocab = create_clean_vocabulary(inverted_vocab)
        output_data = []
        start_time = time.time()

        for test in tests:
            result_dump = process_test(test, functions, model, clean_vocab,
                                       inverted_vocab)
            output_data.append(result_dump)

        save_output(output_data, args.output)
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        print(f"Total execution time: {int(minutes)}m {seconds:.2f}s")

    except KeyboardInterrupt:
        print("\nExecution interrupted by user. Exiting gracefully...")
        sys.exit(1)


if __name__ == "__main__":
    main()
