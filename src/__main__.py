import argparse
from src.data_loader import load_functions, load_tests


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
    print(f"Loaded {len(functions)} functions.")
    print(f"First function: {functions[0].name}")
    print(f"Loaded {len(tests)} tests.")
    print(f"First test: {tests[0].prompt}")


if __name__ == "__main__":
    main()
