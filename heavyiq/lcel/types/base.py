from typing import Literal, TypedDict


class StepDict(TypedDict):
    event: Literal["step", "output", "error"]
    data: str
