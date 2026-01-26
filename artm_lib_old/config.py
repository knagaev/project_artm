# config.py
import spacy
from preprocessing.pipeline import TextPreprocessor
from preprocessing.steps import (
    lowercase,
    make_lemmatizer,
    make_stopwords_remover,
    remove_extra_spaces,
    remove_punctuation,
)
from preprocessing.tokenizer import make_tokenizer

# Загрузка spaCy
nlp = spacy.load("en_core_web_sm")


def create_preprocessor(use_lemmatization: bool = True, custom_stopwords: set = None):
    steps = [lowercase, remove_punctuation, remove_extra_spaces]

    if custom_stopwords:
        steps.append(make_stopwords_remover(custom_stopwords))

    if use_lemmatization:
        steps.append(make_lemmatizer(lambda t: nlp(t)[0].lemma_))

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
