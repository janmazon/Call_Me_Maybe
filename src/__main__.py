import argparse
import numpy as np
from llm_sdk import Small_LLM_Model
from src.data_loader import load_functions, load_tests, load_vocabulary
from src.generator import create_prompt
from src.masking import select_function, get_allowed_tokens_for_args


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

    for test in tests:
        print(f"\n--- Probando: {test.prompt} ---")

        prompt_tokens = create_prompt(functions, test.prompt)
        input_ids = model.encode(prompt_tokens).tolist()[0] 
        chosen_function, final_input_ids = select_function(model, functions, input_ids)
        print(f"Función elegida: {chosen_function.name}")

        current_context_ids = list(final_input_ids)

        json_tokens_generated = []
        max_tokens = 50

        print("Generando JSON: ", end="", flush=True)

        for i in range(max_tokens):
            logits = model.get_logits_from_input_ids(current_context_ids)
            allowed_ids = get_allowed_tokens_for_args(chosen_function, json_tokens_generated, inverted_vocab)
            if not allowed_ids:
                print("\n[Error: El validador bloqueó todos los tokens. Revisar reglas.]")
                break
            mask = np.full_like(logits, -np.inf)
            mask[allowed_ids] = 0.0
            masked_logits = np.array(logits) + mask
            next_token_id = int(np.argmax(masked_logits))
            json_tokens_generated.append(next_token_id)
            current_context_ids.append(next_token_id)
            tecla = inverted_vocab[next_token_id]
            print(tecla, end="", flush=True)
            generated_text = ""
            for num in json_tokens_generated:
                generated_text += inverted_vocab[num]
            if generated_text.endswith("}") and generated_text.count('"') % 2 == 0:
                break
        print("\n--- Test terminado ---")


if __name__ == "__main__":
    main()
