import jax
from tests.test_metrics import vocab_size
import re
from typing import Sequence, Callable, Iterable

import jax.numpy as jnp
from jax import Array
import numpy as np
from numpy.typing import NDArray

from nltk import word_tokenize
from nltk.corpus import stopwords as default_stopwords
from nltk.stem import PorterStemmer


class DatasetPreprocessor:
    def __init__(
            self,
            *,
            lower: bool = True,
            vocabulary: dict | None = None,
            preprocessor: Callable[[str], str] | None = None,
            tokenizer: Callable[[str], list[str]] | None = None,
            stopwords: Iterable[str] | None = None,
            min_word_len: int = 2
    ):
        """
        Convert sequence of raw documents into a sequence of tokens
        suitable for fitting the model or batching via BatchLoader.

        Args:
            lower: convert all characters to lowercase before tokenizing.
            vocabulary: mapping (e.g., a dict) where keys are terms and values
                are unique integers from 0 to len(vocabulary). If not given,
                a vocabulary is determined from the input documents.
            preprocessor: override the preprocessing stage.
            tokenizer: override the tokenizer stage.
            stopwords: terms to be ignored in tokenized data.
        """
        self._lower = lower
        self._vocab = vocabulary
        self._data = None
        self._doc_bounds = None

        self._batch_data: list[tuple[NDArray, NDArray]] = []
        self._batch_data_jax: list[tuple[Array, Array]] = []

        if preprocessor is not None and not callable(preprocessor):
            raise TypeError(
                f'Preprocessor should be callable if provided, '
                f'got type {type(preprocessor)}.'
            )
        self._preprocessor = preprocessor

        if tokenizer is not None and not callable(tokenizer):
            raise TypeError(
                f'Tokenizer should be callable if provided, '
                f'got type {type(tokenizer)}.'
            )
        self._tokenizer = tokenizer

        if stopwords is None:
            self._stopwords = set(default_stopwords.words("english"))
        else:
            try:
                self._stopwords = set(stopwords)
            except TypeError:
                raise
        
        self._min_word_len = min_word_len

    def fit(
            self,
            data: Sequence[str],
    ) -> dict | None:
        """
        Learn a vocabulary dictionary of all tokens in the raw documents.

        Args:
            data: a sequence of strings.
        """
        texts_tokenized = []
        for doc in data:
            texts_tokenized.append(self._preprocess_text(doc))
        self._vocab = self._create_vocabulary(texts_tokenized)
        return self.vocabulary

    def fit_transform(
            self,
            data: Sequence[str],
            *,
            return_doc_bounds: bool = True,
    ) -> Array | tuple[Array, Array]:
        """
        Learn the vocabulary dictionary and return a flattened list of all
        terms from all documents.

        Args:
            data: a sequence of strings.
            return_doc_bounds: if True, returns indices of document bounds
                as the second value (with the first value 0 and the last
                value is len(data)).
        """
        texts_tokenized = []
        for doc in data:
            texts_tokenized.append(self._preprocess_text(doc))

        if self._vocab is None:
            self._vocab = self._create_vocabulary(texts_tokenized)

        self._data = []
        self._doc_bounds = [0, ]
        for text in texts_tokenized:
            self._data.extend([self._vocab[word] for word in text])
            self._doc_bounds.append(len(self._data))

        self._data = jnp.array(self._data, dtype=int)
        self._doc_bounds = jnp.array(self._doc_bounds, dtype=int)

        if return_doc_bounds:
            return self._data, self._doc_bounds
        return self._data

    def fit_transform_batch(
            self,
            data: Sequence[str],
            *,
            batch_size:int = 10000,
    ) -> list[tuple[NDArray, NDArray]]:
        """
        Learn the vocabulary dictionary and return a flattened list of all
        terms from all documents.

        Args:
            data: a sequence of strings.
            return_doc_bounds: if True, returns indices of document bounds
                as the second value (with the first value 0 and the last
                value is len(data)).
        """
        tokens = set()
        texts_tokenized = [[]]
        doc_bounds = [[0]]
        batch_pos = 0
        
        for doc in data:
            doc_tokenized = self._preprocess_text(doc)
            tokens.update(doc_tokenized)
            doc_tokenized_len = len(doc_tokenized)
            if batch_pos + doc_tokenized_len > batch_size:
                texts_tokenized.append([])
                doc_bounds.append([0]) # начинаем с последней границы в последнем батче
                batch_pos = 0
            texts_tokenized[-1].extend(doc_tokenized)
            doc_bounds[-1].append(doc_bounds[-1][-1] + doc_tokenized_len)
            batch_pos += doc_tokenized_len

        if self._vocab is None:
            self._vocab = self._create_vocabulary_from_set(tokens)

        self._batch_data = [(np.array([self._vocab[t] for t in tt]), np.array(db)) 
                                for tt, db 
                                in zip(texts_tokenized, doc_bounds)]

        return self._batch_data


    def fit_transform_batch_jax(
            self,
            data: Sequence[str],
            *,
            batch_size:int = 10000,
    ) -> list[tuple[Array, Array]]:
        """
        Learn the vocabulary dictionary and return a flattened list of all
        terms from all documents.

        Args:
            data: a sequence of strings.
            return_doc_bounds: if True, returns indices of document bounds
                as the second value (with the first value 0 and the last
                value is len(data)).
        """
        tokens = set()
        texts_tokenized = [[]]
        doc_bounds = [[0]]
        batch_pos = 0
        
        for doc in data:
            doc_tokenized = self._preprocess_text(doc)
            tokens.update(doc_tokenized)
            doc_tokenized_len = len(doc_tokenized)

            doc_pos = 0
            while doc_pos < doc_tokenized_len:
                add_tokens = min(doc_tokenized_len - doc_pos, batch_size - batch_pos)
                texts_tokenized[-1].extend(doc_tokenized[doc_pos: doc_pos + add_tokens])
                doc_pos += add_tokens
                doc_bounds[-1].append(doc_bounds[-1][-1] + add_tokens)
                batch_pos += add_tokens
                if batch_pos >= batch_size:
                    texts_tokenized.append([])
                    doc_bounds.append([0])
                    batch_pos = 0

            '''if batch_pos + doc_tokenized_len > batch_size:
                texts_tokenized[-1].extend(doc_tokenized[:(batch_size - batch_pos)])
                texts_tokenized.append(doc_tokenized[:(batch_size - batch_pos)])
                doc_bounds.append([0]) # начинаем с последней границы в последнем батче
                batch_pos = 0
            texts_tokenized[-1].extend(doc_tokenized)
            doc_bounds[-1].append(doc_bounds[-1][-1] + doc_tokenized_len)
            batch_pos += doc_tokenized_len'''

        if self._vocab is None:
            self._vocab = self._create_vocabulary_from_set(tokens)

        self._batch_data_jax = [(jnp.array([self._vocab[t] for t in tt]), jnp.array(db)) 
                                for tt, db 
                                in zip(texts_tokenized, doc_bounds)]

        return self._batch_data_jax

    def _preprocess_text(self, text: str) -> list[str]:
        """Apply preprocessing and tokenization to a single document."""
        # preprocessing stage
        if self._preprocessor is None:
            if self._lower:
                text = text.lower()
                text = re.sub(r'[^a-z]', ' ', text)
            else:
                text = re.sub(r'[^doc_bounds-Za-z]', ' ', text)
        else:
            text = self._preprocessor(text)

        # tokenization stage
        if self._tokenizer is None:
            text_tokenized = word_tokenize(text)
            stemmer = PorterStemmer()
            #text_tokenized = [stemmer.stem(token) for token in text_tokenized]
            text_tokenized = [stemmer.stem(token) for token in text_tokenized 
                                if token not in self._stopwords 
                                    and len(token) >= self._min_word_len]
        else:
            text_tokenized = self._tokenizer(text)

        # removing stopwords
        #text_tokenized = [word for word in text_tokenized if word not in self._stopwords and len(word) > 2]

        return text_tokenized

    @staticmethod
    def _create_vocabulary(texts: list[list[str]]) -> dict:
        """Create vocabulary from all unique terms in tokenized corpus."""
        unique_words = sorted({word for text in texts for word in text}) #// добавил сортировку для повторяемости идентификаторов токенов
        return {word: token for token, word in enumerate(unique_words)}

    @staticmethod
    def _create_vocabulary_from_set(words: set[str]) -> dict:
        """Create vocabulary from all unique terms in tokenized corpus."""
        sorted_words = sorted(words) #// добавил сортировку для повторяемости идентификаторов токенов
        return {word: token for token, word in enumerate(sorted_words)}

    @property
    def vocabulary(self):
        """
        Mapping used for tokenizing terms.
        """
        return self._vocab

    def fit_transform_plsa(
            self,
            data: Sequence[str],
            *,
            return_doc_bounds: bool = True,
    #) -> jax.Array:
    ) -> np.ndarray:
        """
        Learn the vocabulary dictionary and return a list of lists for
        terms from each document.

        Args:
            data: a sequence of strings.
            return_doc_bounds: if True, returns indices of document bounds
                as the second value (with the first value 0 and the last
                value is len(data)).
        """
        texts_tokenized = []
        for doc in data:
            texts_tokenized.append(self._preprocess_text(doc))

        if self._vocab is None:
            self._vocab = self._create_vocabulary(texts_tokenized)

        vocab_size = len(self._vocab)
        print(f"{vocab_size=}")
        docs_data = []
        for doc in data:
            docs_data.append(
                jnp.bincount(
                    jnp.array([self._vocab[word] for word in self._preprocess_text(doc)]),
                    length=vocab_size,
                    minlength=vocab_size,
            ))

        self._data = np.array(docs_data)

        return self._data


class BatchLoader:
    def __init__(
            self,
            data: Array,
            doc_bounds: Array,
            *,
            batch_size: int = 10000
    ):
        """
        Split tokenized data into batches. Instance of this class can be passed
        directly to ContextTopicModel for batched fitting.

        Args:
            data: array of tokens with shape (I, ),
                where I is total number of words in corpus.
            doc_bounds: array of shape (B, ),
                containing indices of document bounds.
            batch_size: size of a single batch.
        """
        self.batch_size = batch_size
        self._batches = []

        num_batches = jnp.ceil(len(data) / batch_size).astype(int)
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = (i + 1) * self.batch_size
            end_idx = min(end_idx, len(data))

            data_batch = data[start_idx:end_idx]
            bounds_batch_mask = (doc_bounds >= start_idx) & (doc_bounds < end_idx)
            doc_bounds_batch = doc_bounds[bounds_batch_mask].copy()
            doc_bounds_batch -= start_idx  # absolute bounds to batch-relative bounds

            # add bounds at the beginning and ending of the batch
            if len(doc_bounds_batch) == 0 or doc_bounds_batch[0] != 0:
                doc_bounds_batch = jnp.concatenate([
                    jnp.array([0]),
                    doc_bounds_batch,
                ], dtype=int)
            if doc_bounds_batch[-1] != self.batch_size:
                doc_bounds_batch = jnp.concatenate([
                    doc_bounds_batch,
                    jnp.array([end_idx - start_idx]),
                ], dtype=int)

            self._batches.append((data_batch, doc_bounds_batch))
        '''
        if len(doc_bounds) == 0:
            return []

        split_idx = []
        start = 0
        while start < len(doc_bounds):
            # Ищем первый индекс, где значение > doc_bounds[start] + max_dist
            end = np.searchsorted(doc_bounds, doc_bounds[start] + self.batch_size, side='right')
            split_idx.append(end)
            start = end

        # np.split принимает индексы разрезов (последний len(doc_bounds) отбрасываем)
        return np.split(doc_bounds, split_idx[:-1])
        '''

    def __len__(self):
        return len(self._batches)

    def __getitem__(self, idx) -> tuple[Array, Array]:
        return self._batches[idx]
