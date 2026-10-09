from pydantic import BaseModel, Field, field_validator
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

    @field_validator("parameters")
    @classmethod
    def validate_parameters_names(cls, value: dict[str, ParameterDefinition]
                                  ) -> dict[str, ParameterDefinition]:
        """
        Checks that parameter names are not empty.

        Makes sure the keys in the parameters dictionary have real text
        and are not just empty strings or spaces.

        Args:
            value (dict[str, ParameterDefinition]):
            The parameters dictionary to check.

        Returns:
            dict[str, ParameterDefinition]: The checked parameters dictionary.

        Raises:
            ValueError: If any key is empty or only has spaces.
        """

        for key in value.keys():
            if not key.strip():
                raise ValueError
        return value


class TestPrompt(BaseModel):
    """Model to represent a user test prompt."""

    prompt: str = Field(..., min_length=1)


class FunctionCallResult(BaseModel):
    """
    Model to structure the final generated result.
    It makes sure the final output has exactly the required keys.
    """

    prompt: str
    name: str
    parameters: dict[str, Any]
