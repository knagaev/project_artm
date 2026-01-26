# preprocessing/steps.py
import re
from typing import Callable


# --- Базовые этапы ---
def lowercase(text: str) -> str:
    return text.lower()


def remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text)


def remove_extra_spaces(text: str) -> str:
    return " ".join(text.split())


# --- Стоп-слова ---
def make_stopwords_remover(stopwords: set[str]) -> Callable[[str], str]:
    def remove_stopwords(text: str) -> str:
        tokens = text.split()
        filtered = [t for t in tokens if t not in stopwords]
        return " ".join(filtered)

    return remove_stopwords


# --- Лемматизация ---
def make_lemmatizer(lemmatizer_func: Callable[[str], str]) -> Callable[[str], str]:
    def lemmatize_text(text: str) -> str:
        tokens = text.split()
        lemmatized = [lemmatizer_func(t) for t in tokens]
        return " ".join(lemmatized)

    return lemmatize_text
