from src.models import FunctionDefinition


def create_prompt(functions: list[FunctionDefinition], prompt: str) -> str:
    text = "Available functions:\n"
    for function in functions:
        text += f"- {function.name}: {function.description}\n"
    text += f"\nUser question: {prompt}\nSelected function:"

    return text
