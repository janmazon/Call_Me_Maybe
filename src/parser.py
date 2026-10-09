import argparse


def parse_arguments() -> argparse.Namespace:
    """
    Sets up and reads command line arguments.

    It handles the paths for input files (functions and tests) and the
    output file. It includes default values to make it easy to use.

    Returns:
        argparse.Namespace: An object containing the read arguments.
    """

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
