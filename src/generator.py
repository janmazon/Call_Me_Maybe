from src.models import FunctionDefinition
import json


def create_prompt(functions: list[FunctionDefinition], prompt: str) -> str:
    """
    Creates the final formatted text to send to the model.

    It uses the model's special chat format to include system instructions,
    available functions in JSON format, and the user's message.

    Args:
        functions (list[FunctionDefinition]): List of available functions.
        prompt (str): The text written by the user.

    Returns:
        str: The full text formatted with special tokens.
    """

    functions_dict = []
    for f in functions:
        functions_dict.append(f.model_dump())
    functions_text = json.dumps(functions_dict, indent=2)
    instructions = ("Extract the parameters from the user's request and "
                    "output a valid JSON object. Match the function "
                    "signature exactly. For regular expressions, use "
                    "pure Python syntax.")
    final_prompt = (
        f"<|im_start|>system\n{instructions}\n\n"
        f"Available functions:\n{functions_text}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return final_prompt
