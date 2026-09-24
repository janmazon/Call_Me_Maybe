import argparse
from llm_sdk import Small_LLM_Model
from src.data_loader import load_functions, load_tests
from src.generator import create_prompt
from src.masking import select_function


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
    for test in tests:
        prompt = create_prompt(functions, test.prompt)
        raw_input_ids = model.encode(prompt).tolist()
        if raw_input_ids and isinstance(raw_input_ids[0], list):
            input_ids = raw_input_ids[0]
        else:
            input_ids = raw_input_ids
        selected, final_ids = select_function(model, functions, input_ids)
        print(f"Pregunta: {test.prompt}")
        print(f"Función elegida: {selected.name}\n")


if __name__ == "__main__":
    main()
