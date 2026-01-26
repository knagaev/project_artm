# preprocessing/tokenizer.py
from typing import Callable, Optional

from .pipeline import TextPreprocessor


def make_tokenizer(
    preprocessor: Optional[TextPreprocessor] = None, min_token_length: int = 2
) -> Callable[[str], list[str]]:
    """
    Создаёт токенизатор с предобработкой.

    Args:
        preprocessor: цепочка предобработки текста
        min_token_length: минимальная длина токена
    """

    def tokenizer(text: str) -> list[str]:
        if preprocessor is not None:
            text = preprocessor(text)
        tokens = text.split()
        # Фильтрация коротких токенов и чисел (опционально)
        tokens = [t for t in tokens if len(t) >= min_token_length and not t.isdigit()]
        return tokens

    return tokenizer
