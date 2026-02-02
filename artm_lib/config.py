# config.py
# import spacy

# import nltk
# from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from artm_lib.preprocessing.pipeline import TextPreprocessor
from artm_lib.preprocessing.steps import (
    lowercase,
    make_lemmatizer,
    make_stopwords_remover,
    remove_extra_spaces,
    remove_punctuation,
)
from artm_lib.preprocessing.tokenizer import make_tokenizer

from .preprocessing.tokenizer import create_spacy_tokenizer
from spacy.lang.en import stop_words as SPACY_ENGLISH_STOPWORDS

# Загрузите данные один раз
# nltk.download("wordnet")
# nltk.download("omw-1.4")  # для поддержки других языков

# Стоп-слова (можно расширить)
"""ENGLISH_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
}"""

# Создание токенизатора
spacy_tokenizer = create_spacy_tokenizer(
    stopwords=SPACY_ENGLISH_STOPWORDS.STOP_WORDS,
    min_token_length=2,
    disable_pipes=["parser", "ner", "textcat"],  # ускорение
)

"""
lemmatizer = WordNetLemmatizer()


def lemmatize_word(word: str) -> str:
    return lemmatizer.lemmatize(word.lower())

def create_preprocessor(use_lemmatization: bool = True, custom_stopwords: set | None = None):
    steps = [lowercase, remove_punctuation, remove_extra_spaces]

    if custom_stopwords:
        steps.append(make_stopwords_remover(custom_stopwords))

    if use_lemmatization:
        steps.append(make_lemmatizer(lambda t: lemmatize_word(t)))

    return TextPreprocessor(steps)


# Создание токенизатора
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
}

preprocessor = create_preprocessor(use_lemmatization=True, custom_stopwords=STOPWORDS)

tokenizer = make_tokenizer(preprocessor=preprocessor, min_token_length=2)
"""
