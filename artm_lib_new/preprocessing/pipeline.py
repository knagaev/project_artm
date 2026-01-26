# preprocessing/pipeline.py
from typing import Callable


class TextPreprocessor:
    def __init__(self, steps: list[Callable[[str], str]]):
        self.steps = steps

    def __call__(self, text: str) -> str:
        for step in self.steps:
            text = step(text)
        return text
