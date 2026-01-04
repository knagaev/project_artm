"""
# Полный пример: от данных до обучения
import torch

from artm_lib.data.dataset import ARTMDatasetParquet
from artm_lib.data.collators import ARTMCollator
from artm_lib.artm.model import ARTM
from artm_lib.preprocessing.vocabulary import build_vocab_and_index_from_parquet
from artm_lib.preprocessing.tokenizer import simple_tokenizer


def main():
    token_to_id, doc_index = build_vocab_and_index_from_parquet("data/pq")
    dataset = ARTMDatasetParquet(doc_index, token_to_id, "text", simple_tokenizer)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        collate_fn=ARTMCollator(len(token_to_id)),
        num_workers=4,
    )

    model = ARTM(n_topics=20, vocab_size=len(token_to_id))
    for doc_ids, bow in loader:
        model.fit_batch(bow)
"""

# examples/train_artm.py
from torch.utils.data import DataLoader

# from artm_lib.artm.model import ARTM
from artm_lib.artm.model2 import FullEM_ARTM
from artm_lib.data.collators import ARTMCollator
from artm_lib.data.dataset import ARTMDatasetParquet
from artm_lib.preprocessing.tokenizer import simple_tokenizer
from artm_lib.preprocessing.vocabulary import build_vocab_and_index_from_parquet


def main():
    # 1. Загрузка словаря и данных
    token_to_id, doc_index = build_vocab_and_index_from_parquet(
        parquet_dir_str=r".\artm_lib\data\parquets", tokenizer=simple_tokenizer, min_df=5
    )
    vocab = [""] * len(token_to_id)
    for token, idx in token_to_id.items():
        vocab[idx] = token

    # 2. DataLoader
    dataset = ARTMDatasetParquet(
        doc_index=doc_index,
        token_to_id=token_to_id,
        text_column="Description",
        tokenizer=simple_tokenizer,
    )
    loader = DataLoader(
        dataset, batch_size=256, collate_fn=ARTMCollator(len(token_to_id)), num_workers=4
    )

    # 3. Обучение
    """model = ARTM(
        n_topics=20,
        vocab_size=len(token_to_id),
        alpha=0.1,
        beta=0.01,
        sparsity_phi=0.1,
        random_state=42,
    )"""
    model = FullEM_ARTM(
        n_topics=10,
        vocab_size=len(token_to_id),
        random_state=42,
    )
    model.fit(loader, n_iter=5)

    # 4. Вывод тем
    top_words = model.get_top_words(vocab, top_n=10)
    for i, words in enumerate(top_words):
        print(f"Topic {i}: {' '.join(words)}")


if __name__ == "__main__":
    main()
