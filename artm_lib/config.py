# config.py
# import spacy

import nltk
from nltk.corpus import wordnet
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

# Загрузите данные один раз
# nltk.download("wordnet")
# nltk.download("omw-1.4")  # для поддержки других языков

lemmatizer = WordNetLemmatizer()


def lemmatize_word(word: str) -> str:
    return lemmatizer.lemmatize(word.lower())


"""
# Загрузка spaCy
nlp = spacy.load("en_core_web_sm")

def create_preprocessor(use_lemmatization: bool = True, custom_stopwords: set | None = None):
    steps = [lowercase, remove_punctuation, remove_extra_spaces]

    if custom_stopwords:
        steps.append(make_stopwords_remover(custom_stopwords))

    if use_lemmatization:
        steps.append(make_lemmatizer(lambda t: nlp(t)[0].lemma_))

    return TextPreprocessor(steps)
"""


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
