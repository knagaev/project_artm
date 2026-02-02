# artm_lib/preprocessing/russian_utils.py
"""
Russian text preprocessing with spaCy + pymorphy2.
"""

import re
from functools import lru_cache
from typing import Callable, Optional

import pymorphy2
import spacy

# Инициализация анализаторов (один раз при импорте)
_nlp = spacy.load("ru_core_news_sm")
_morph = pymorphy2.MorphAnalyzer()

# Предкомпилированные регулярки
_CYRILLIC_PATTERN = re.compile(r"^[а-яё]+$")


# Кэширование лемматизации
@lru_cache(maxsize=200000)
def _lemmatize_russian_token(token: str) -> str:
    """Лемматизация русского токена с кэшированием."""
    parsed = _morph.parse(token)[0]
    return parsed.normal_form


def create_russian_tokenizer(
    stopwords: Optional[set[str]] = None,
    min_token_length: int = 2,
    disable_pipes: list[str] = ["parser", "ner"],
) -> Callable[[str], list[str]]:
    """
    Создаёт токенизатор для русского языка.

    Использует spaCy для токенизации и pymorphy2 для лемматизации.
    """
    # Загрузка spaCy без ненужных компонентов
    nlp = spacy.load("ru_core_news_sm", disable=disable_pipes)

    def tokenize(text: str) -> list[str]:
        if not text or not isinstance(text, str):
            return []

        # Приведение к нижнему регистру и базовая очистка
        text = text.lower()
        # Удаление цифр и специальных символов (оставляем только кириллицу и пробелы)
        text = re.sub(r"[^а-яё\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return []

        # Токенизация через spaCy
        doc = nlp(text)

        tokens = []
        for token in doc:
            word = token.text.strip()

            # Фильтрация
            if (
                word
                and len(word) >= min_token_length
                and _CYRILLIC_PATTERN.match(word)  # только кириллица
                and not token.is_space
                and not token.is_punct
            ):
                # Лемматизация через pymorphy2
                lemma = _lemmatize_russian_token(word)

                # Фильтрация стоп-слов
                if stopwords and lemma in stopwords:
                    continue

                tokens.append(lemma)

        return tokens

    return tokenize
