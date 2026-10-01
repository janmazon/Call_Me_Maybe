from src.models import FunctionDefinition


def create_prompt(functions: list[FunctionDefinition], prompt: str) -> str:
    text = (
        "You are an expert AI assistant that extracts arguments "
        "for function calls.\n"
        "You must respond ONLY with a valid JSON object.\n"
        "The JSON must have exactly this structure: "
        '{"name": "function_name", "parameters": {"arg1": value}}\n\n'
        "CRITICAL RULES:\n"
        "1. Numbers MUST be written as float with a decimal point"
        "(e.g., 2.0, 144.0).\n"
        "2. Strings MUST NOT cointain extra quotes, colons, or punctuation."
        "(e.g., write hello, not 'hello').\n\n"
        "Available functions:\n"
    )
    for function in functions:
        text += f"- {function.name}: {function.description}\n"
        text += f"  Parameters: {function.parameters}\n"
    text += f"\nUser question: {prompt}\n"
    text += "JSON Response:"

    return text
