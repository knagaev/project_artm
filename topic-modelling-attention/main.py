from sys import path as syspath
from os import path as ospath
import jax.numpy as jnp

from sklearn.datasets import fetch_20newsgroups

import matplotlib.pyplot as plt
import seaborn as sns

#from cartm.experimental_model import ExperimentalContextTopicModel
#from cartm.attentive_model import AttentiveTopicModel
#from cartm.improved_model import AttentiveTopicModel
from cartm.attentive_model import MatveyAttentiveTopicModel

from cartm.preprocessing import DatasetPreprocessor
from time import time

if __name__ == '__main__':

    data = fetch_20newsgroups(data_home='./data/', subset='all').data
    #data = data[:400]
    #with open('./data/test_data.txt') as f:
    #    data = f.readlines()

    preprocessor = DatasetPreprocessor()
    tokenized_data, document_bounds = preprocessor.fit_transform(data)

    attentive_topic_model = MatveyAttentiveTopicModel(
        vocab_size=len(preprocessor.vocabulary),
        ctx_len=10,
        n_topics=10
    )

    attentive_topic_model.fit(
        data=tokenized_data,
        ctx_bounds=document_bounds,
        verbose=2,
        seed=42,
    )

    print(attentive_topic_model.phi[0])

    '''
    for i in range(5):
        start_time = time()
        attentive_topic_model.fit(
            data=tokenized_data,
            ctx_bounds=document_bounds,
            verbose=2,
            seed=42,
        )
        end_time = time()
        print(f"attentive_topic_model {end_time - start_time}")
    def test():
        attentive_topic_model.fit(
            data=tokenized_data,
            ctx_bounds=document_bounds,
            verbose=2,
            seed=42,
        )


    import timeit
    n = 10
    print(timeit.timeit(stmt="test()", number=n, setup="from __main__ import test") / n)'''