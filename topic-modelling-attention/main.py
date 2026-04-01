from sys import path as syspath
from os import path as ospath
import jax.numpy as jnp

from sklearn.datasets import fetch_20newsgroups

import matplotlib.pyplot as plt
import seaborn as sns

syspath.insert(1, r'D:\ARTM\project_artm\topic-modelling-attention')

#from cartm.experimental_model import ExperimentalContextTopicModel
from cartm.attentive_model import AttentiveTopicModel

from cartm.preprocessing import DatasetPreprocessor
from time import time

data = fetch_20newsgroups(data_home='./data/', subset='all').data
#with open('./data/test_data.txt') as f:
#    data = f.readlines()

preprocessor = DatasetPreprocessor()
tokenized_data, document_bounds = preprocessor.fit_transform(data)

attentive_topic_model = AttentiveTopicModel(
    vocab_size=len(preprocessor.vocabulary),
    ctx_len=10,
    n_topics=10
)

start_time = time()
attentive_topic_model.fit(
    data=tokenized_data,
    ctx_bounds=document_bounds,
    verbose=2,
    seed=42,
)
end_time = time()
print(f"attentive_topic_model {end_time - start_time}")