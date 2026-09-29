import argparse
import json
import os
import numpy as np
from llm_sdk import Small_LLM_Model
from src.data_loader import load_functions, load_tests, load_vocabulary
from src.data_loader import create_clean_vocabulary
from src.generator import create_prompt
from src.masking import select_function, get_allowed_tokens_for_args
from src.models import FunctionCallResult


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM function calling tool "
                                     "using constrained decoding")
    parser.add_argument("--functions_definition",
                        type=str,
                        default="data/input/functions_definition.json",
                        help="Path to the JSON file containing "
                        "function definitions")
    parser.add_argument("--input",
                        type=str,
                        default="data/input/function_calling_tests.json",
                        help="Path to the JSON file containing "
                        "input test prompts")
    parser.add_argument("--output",
                        type=str,
                        default="data/output/function_calling_results.json",
                        help="Path to the JSON file where results "
                        "will be saved")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    functions = load_functions(args.functions_definition)
    tests = load_tests(args.input)
    model = Small_LLM_Model()
    inverted_vocab = load_vocabulary(model)
    clean_vocab = create_clean_vocabulary(inverted_vocab)
    output_data = []

    for test in tests:
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
        result = FunctionCallResult(prompt=test.prompt,
                                    name=chosen_function.name,
                                    parameters=json_data)
        output_data.append(result.model_dump())
    except json.JSONDecodeError:
        print(f"Error parsinng generated JSON for prompt: {test.prompt}.")
        print(f"Problematic text: {generated_text}")
    except KeyError as e:
        print(f"Missing key {e} in JSON for prompt: {test.prompt}.")

    dir = os.path.dirname(args.output)
    if dir:
        os.makedirs(dir, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    main()
