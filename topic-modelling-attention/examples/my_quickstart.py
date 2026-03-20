from sys import path as syspath
from os import path as ospath

syspath.insert(1, r'D:\ARTM\topic-modelling-attention\src')

import jax.numpy as jnp

from sklearn.datasets import fetch_20newsgroups

import matplotlib.pyplot as plt
import seaborn as sns

from cartm.model import ContextTopicModel
from cartm.preprocessing import DatasetPreprocessor


data = fetch_20newsgroups(data_home='./data/', subset='all').data
print(f'Total number of documents in corpus: {len(data)}')
print(f'Total number of words in corpus: {sum([len(doc.split(" ")) for doc in data])}')
print(f'Total number of words in corpus: {sum([len(doc.split()) for doc in data])}')

preprocessor = DatasetPreprocessor()
tokenized_data, document_bounds = preprocessor.fit_transform(data)

topic_model = ContextTopicModel(
    vocab_size=len(preprocessor.vocabulary) if preprocessor.vocabulary is not None else 0,
    ctx_len=10,
    n_topics=20
)

topic_model.fit(
    data=tokenized_data,
    ctx_bounds=document_bounds,
    verbose=2,
    seed=42,
)