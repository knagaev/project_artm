import jax.numpy as jnp
import numpy as np

from sklearn.datasets import fetch_20newsgroups

#from cartm.experimental_model import ExperimentalContextTopicModel
#from cartm.attentive_model import AttentiveTopicModel
#from cartm.improved_model import AttentiveTopicModel
#from cartm.attentive_model import MatveyAttentiveTopicModel
from cartm.my_attentive_model import MyAttentiveTopicModel

from cartm.preprocessing import DatasetPreprocessor
from time import time

import warnings
warnings.simplefilter('error', RuntimeWarning)

if __name__ == '__main__':

    data = fetch_20newsgroups(data_home='./data/', subset='all').data
    data = data[:400]
    #with open('./data/test_data.txt') as f:
    #    data = f.readlines()

    preprocessor = DatasetPreprocessor(
        stopwords=None, #set(), 
        min_word_len=2
        )

    #tokenized_data, document_bounds = preprocessor.fit_transform(data)
    batch_data = preprocessor.fit_transform_batch_jax(data, batch_size=1000)
    #batch_data = preprocessor.fit_transform_batch_jax(data, batch_size=20)
    #print(batch_data)
    print(len(preprocessor.vocabulary)) 

    attentive_topic_model = MyAttentiveTopicModel(
        vocab_size=len(preprocessor.vocabulary),
        ctx_len=3,
        n_topics=10
    )

    attentive_topic_model.fit_jax(
        batch_data_with_doc_bounds=batch_data,
        seed=42,
    )

    print(attentive_topic_model.phi[:, 0])

    np.save('my_phi.npy', np.asarray(attentive_topic_model.phi))

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