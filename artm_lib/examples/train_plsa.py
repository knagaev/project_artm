from scipy.sparse import vstack

from artm_lib.config import tokenizer
from artm_lib.data.collators import ARTMCollator
from artm_lib.data.dataset import ARTMDatasetParquet

# from torch.utils.data import DataLoader
from artm_lib.data.simple_loader import SimpleDataLoader
from artm_lib.plsa.model import PLSA

# from artm_lib.preprocessing.tokenizer import simple_tokenizer
from artm_lib.preprocessing.vocabulary import build_vocab_and_index_from_parquet

# 1. Загрузка данных
parquet_dir = r".\artm_lib\data\parquets"
"""
token_to_id, doc_index = build_vocab_and_index_from_parquet(
    parquet_dir_str=parquet_dir, tokenizer=simple_tokenizer, min_df=5
)
"""
print("Загрузка данных...")
token_to_id, doc_index = build_vocab_and_index_from_parquet(
    parquet_dir_str=parquet_dir,
    tokenizer=tokenizer,
    max_df=0.95,
)
V = len(token_to_id)

# 2. DataLoader
dataset = ARTMDatasetParquet(
    doc_index=doc_index,
    token_to_id=token_to_id,
    text_column="Description",
    tokenizer=tokenizer,
)

loader_batch_size = 1000
"""
loader = DataLoader(
    dataset, batch_size=loader_batch_size, collate_fn=ARTMCollator(V), num_workers=0
)
"""

loader = SimpleDataLoader(
    dataset,
    collate_fn=ARTMCollator(V),
    batch_size=loader_batch_size,
    shuffle=True,  # работает!
    seed=42,  # опционально, для воспроизводимости
)

# 3. Сбор полной матрицы
print("Сбор матрицы...")
all_bows = []
dataset_size = len(dataset)
print(f"Size of dataset: {dataset_size}")
num_batches = -(-dataset_size // loader_batch_size)
print(f"Number o batches: {num_batches}")
i = 0
for _, bow in loader:
    all_bows.append(bow)
    i += 1
    print(f"Batch #{i} of {num_batches} loaded")
X = vstack(all_bows)

# 4. Обучение
model = PLSA(n_topics=20, vocab_size=V, random_state=42)
model.fit_full(X, max_iter=100)

# 5. Оценка
vocab = [""] * V
for token, idx in token_to_id.items():
    vocab[idx] = token

top_words = model.get_top_words(vocab, top_n=10)
for i, words in enumerate(top_words):
    print(f"Тема {i}: {words}")

perplexity = model.score_perplexity_from_matrix(X)  # новый метод
print(f"Perplexity: {perplexity:.2f}")
