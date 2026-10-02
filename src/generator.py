from src.models import FunctionDefinition


def create_prompt(functions: list[FunctionDefinition], prompt: str) -> str:
    text = (
        "You are an expert AI assistant that "
        "extracts arguments for function calls.\n"
        "You must output the function name immediately "
        "followed by a JSON object with the parameters.\n\n"
        "CRITICAL RULES:\n"
        "1. Numbers MUST be written as floats with a decimal point.\n"
        "2. Strings MUST NOT contain surrounding single quotes. "
        "Remove them if present.\n"
        "3. Regular expressions must be standard Python format, "
        "without surrounding slashes (e.g. \\d+, not /\\d+/g).\n"
        "4. Pay close attention to exact words requested and plurals.\n\n"
        "EXAMPLES:\n"
        "Question: Reverse the string 'testing'\n"
        'FunctionCall: fn_reverse_string{"s":"testing"}\n\n'
        'Question: Replace all numbers in "I have 2 cats" with DIGITS\n'
        'FunctionCall: fn_substitute_string_with_regex{"source_string":"I '
        'have 2 cats","regex":"\\\\d+","replacement":"DIGITS"}\n\n'
        "Question: Greet batman\n"
        'FunctionCall: fn_greet{"name":"batman"}\n\n'
        "Available functions:\n"
    )

    for function in functions:
        text += f"- {function.name}: {function.description}\n"
        text += f"  Parameters: {function.parameters}\n"
    text += f"\nUser question: {prompt}\n"
    text += "FunctionCall: "

    return text
