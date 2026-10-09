from pydantic import BaseModel
from typing import Any


class ParameterDefinition(BaseModel):
    """Model to represent a single parameter definition."""

    type: str


class ReturnDefinition(BaseModel):
    """Model to represent a function's return definition."""

    type: str


class FunctionDefinition(BaseModel):
    """
    Full model for an available function.
    It includes its name, description, parameters, and return type.
    """

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition


class TestPrompt(BaseModel):
    """Model to represent a user test prompt."""

    prompt: str


class FunctionCallResult(BaseModel):
    """
    Model to structure the final generated result.
    It makes sure the final output has exactly the required keys.
    """

    prompt: str
    name: str
    parameters: dict[str, Any]
