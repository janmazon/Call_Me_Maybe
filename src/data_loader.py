from src.models import FunctionDefinition, TestPrompt
import json
import sys


def load_functions(file_path: str) -> list[FunctionDefinition]:
    try:
        with open(file_path, "r") as f:
            content = json.load(f)
            functions: list[FunctionDefinition] = []
            if isinstance(content, list):
                for i in content:
                    functions.append(FunctionDefinition.model_validate(i))
            else:
                print(f"Error: Expected a list in '{file_path}'.")
                sys.exit(1)

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' contains invalid JSON.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Unexpected error while loading '{file_path}': {e}")
        sys.exit(1)

    return functions


def load_tests(file_path: str) -> list[TestPrompt]:
    try:
        with open(file_path, "r") as f:
            content = json.load(f)
            prompts: list[TestPrompt] = []
            if isinstance(content, list):
                for i in content:
                    prompts.append(TestPrompt.model_validate(i))
            else:
                print(f"Error: Expected a list in '{file_path}'.")
                sys.exit(1)

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' contains invalid JSON.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Unexpected error while loading '{file_path}': {e}")
        sys.exit(1)

    return prompts
