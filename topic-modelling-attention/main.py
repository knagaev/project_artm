from sys import path as syspath
from os import path as ospath
import jax.numpy as jnp

from sklearn.datasets import fetch_20newsgroups

import matplotlib.pyplot as plt
import seaborn as sns

from cartm.model import ContextTopicModel
from cartm.preprocessing import DatasetPreprocessor

syspath.insert(1, r'D:\ARTM\topic-modelling-attention\src')

#data = fetch_20newsgroups(data_home='./data/', subset='all').data
with open('./data/test_data.txt') as f:
    data = f.readlines()

preprocessor = DatasetPreprocessor()
tokenized_data, document_bounds = preprocessor.fit_transform(data)

topic_model = ContextTopicModel(
    vocab_size=len(preprocessor.vocabulary),
    ctx_len=10,
    n_topics=10
)

topic_model.fit(
    data=tokenized_data,
    ctx_bounds=document_bounds,
    verbose=2,
    seed=42,
)