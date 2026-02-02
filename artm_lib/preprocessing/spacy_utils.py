# artm_lib/preprocessing/spacy_utils.py
"""
spaCy-based text preprocessing utilities.
"""

from functools import lru_cache
from typing import Callable, Optional

import spacy

# Загрузка модели один раз при импорте
_nlp = spacy.load("en_core_web_sm")


# Кэширование для ускорения лемматизации
@lru_cache(maxsize=100000)
def _lemmatize_token(token: str) -> str:
    """Лемматизирует отдельный токен с кэшированием."""
    return _nlp(token)[0].lemma_.lower()


def create_spacy_tokenizer(
    stopwords: Optional[set[str]] = None,
    min_token_length: int = 2,
    disable_pipes: list[str] = ["parser", "ner"],
) -> Callable[[str], list[str]]:
    """
    Создаёт токенизатор на основе spaCy с лемматизацией.

    Args:
        stopwords: множество стоп-слов для фильтрации
        min_token_length: минимальная длина токена
        disable_pipes: компоненты spaCy, которые можно отключить для скорости

    Returns:
        Функция токенизации: str → list[str]
    """
    # Создаём отдельный pipeline без ненужных компонентов
    nlp = spacy.load("en_core_web_sm", disable=disable_pipes)

    def tokenize(text: str) -> list[str]:
        if not text or not isinstance(text, str):
            return []

        # Обработка текста через spaCy
        doc = nlp(text)

        tokens = []
        for token in doc:
            # Фильтрация
            if (
                token.is_alpha  # только буквы
                and not token.is_space  # не пробелы
                and not token.is_punct  # не пунктуация
                and len(token.text) >= min_token_length
            ):
                lemma = token.lemma_.lower()

                # Фильтрация стоп-слов
                if stopwords and lemma in stopwords:
                    continue

                tokens.append(lemma)

        return tokens

    return tokenize
