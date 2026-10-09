from src.models import FunctionDefinition, TestPrompt
from llm_sdk.llm_sdk import Small_LLM_Model
import json
import sys
import string


def load_functions(file_path: str) -> list[FunctionDefinition]:
    """
    Loads and checks function definitions from a JSON file.

    Args:
        file_path (str): The path to the JSON file with the definitions.

    Returns:
        list[FunctionDefinition]: A list of checked function objects.

    Raises:
        SystemExit: If the file is missing, the JSON is bad, required fields
        are missing, or data types are not allowed.
    """

    try:
        with open(file_path, "r") as f:
            content = json.load(f)
            functions: list[FunctionDefinition] = []
            if isinstance(content, list):
                for i in content:
                    functions.append(FunctionDefinition.model_validate(i))
                if not functions:
                    print(f"Error: The functions array in '{file_path}' "
                          f"is empty.")
                    sys.exit(1)
                allowed_types = ["string", "integer", "number",
                                 "bool", "boolean"]
                for function in functions:
                    for param_name, param_def in function.parameters.items():
                        if param_def.type not in allowed_types:
                            print(f"Error: Function '{function.name}' has an "
                                  f"unsupported parameter type "
                                  f"'{param_def.type}' for '{param_name}'. \n"
                                  f"Allowed types are: {allowed_types}.")
                            sys.exit(1)
            else:
                print(f"Error: Expected a list in '{file_path}'.")
                sys.exit(1)

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' contains invalid JSON.")
        sys.exit(1)
    except IsADirectoryError:
        print(f"Error: '{file_path}' is a directory, not a file.")
        sys.exit(1)
    except ValueError:
        print(f"Error: Invalid data format in '{file_path}'. "
              f"Missing or invorrect fields.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Unexpected error while loading '{file_path}': {e}")
        sys.exit(1)

    return functions


def load_tests(file_path: str) -> list[TestPrompt]:
    """
    Loads and checks user test prompts from a JSON file.

    Args:
        file_path (str): The path to the JSON file with the user tests.

    Returns:
        list[TestPrompt]: A list of checked test objects.

    Raises:
        SystemExit: If the file is missing, is a folder, or the JSON format
        is wrong.
    """

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
    except IsADirectoryError:
        print(f"Error: '{file_path}' is a directory, not a file.")
        sys.exit(1)
    except ValueError:
        print(f"Error: Invalid data format in '{file_path}'. "
              f"Missing or invorrect fields.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Unexpected error while loading '{file_path}': {e}")
        sys.exit(1)

    return prompts


def load_vocabulary(model: Small_LLM_Model) -> dict[int, str]:
    """
    Loads the model's vocabulary and makes a reversed dictionary.

    Args:
        model (Small_LLM_Model): The model object from the SDK.

    Returns:
        dict[int, str]: A dictionary where keys are token IDs and values
        are strings.
    """

    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r") as f:
        vocab = json.load(f)

        inverted_vocab: dict[int, str] = {}
        for string_token, num_id in vocab.items():
            inverted_vocab[num_id] = string_token

    return inverted_vocab


def create_clean_vocabulary(inverted_vocab: dict[int, str]) -> dict[int, str]:
    """
    Filters the vocabulary to keep only safe and allowed tokens.

    It keeps only basic text, numbers, punctuation, and spaces. This stops
    the model from using bad characters like emojis or other languages.

    Args:
        inverted_vocab (dict[int, str]): The original reversed vocabulary.

    Returns:
        dict[int, str]: The cleaned vocabulary.
    """

    allowed_chars: str = (string.ascii_letters + string.digits +
                          string.punctuation + " \n\t" + "Ġ" + "Ċ")
    clean_vocab: dict[int, str] = {}
    for key, token in inverted_vocab.items():
        if all(char in allowed_chars for char in token):
            clean_vocab[key] = token

    return clean_vocab
