from scipy.sparse import vstack

from artm_lib.config import spacy_tokenizer as tokenizer  # spaCy-токенизатор

# from artm_lib.config import tokenizer
from artm_lib.data.collators import ARTMCollator
from artm_lib.data.dataset import ARTMDatasetParquet

# from torch.utils.data import DataLoader
from artm_lib.data.simple_loader import SimpleDataLoader
from artm_lib.plsa.model import PLSA

# from artm_lib.preprocessing.tokenizer import simple_tokenizer
from artm_lib.preprocessing.vocabulary import build_vocab_and_index_from_parquet

from sklearn.model_selection import train_test_split

"""
from artm_lib.data.simple_loader import SimpleDataLoader
from artm_lib.data.dataset import ARTMDatasetParquet
from artm_lib.data.collators import ARTMCollator
from artm_lib.plsa.model import PLSA

# 1. Подготовка данных
doc_index = [
    ("data/part_0.parquet", 0),
    ("data/part_0.parquet", 1),
    # ...
]

token_to_id = {"word1": 0, "word2": 1, ...}  # ваш словарь

def simple_tokenizer(text: str) -> list[str]:
    return text.lower().split()

dataset = ARTMDatasetParquet(
    doc_index=doc_index,
    token_to_id=token_to_id,
    text_column="text",
    tokenizer=simple_tokenizer,
    cache_dataframes=True,
)
"""

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


# 2. Разделение на train/val
indices = list(range(len(dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)

# Создаем "поддатасеты" через list comprehension
train_data = [dataset[i] for i in train_idx]
val_data = [dataset[i] for i in val_idx]

# 3. Создаем DataLoader'ы
vocab_size = len(token_to_id)

train_loader = SimpleDataLoader(
    dataset=train_data,
    collate_fn=ARTMCollator(vocab_size),
    batch_size=128,
    shuffle=True,
    seed=42,
)

val_loader = SimpleDataLoader(
    dataset=val_data,
    collate_fn=ARTMCollator(vocab_size),
    batch_size=256,
    shuffle=False,
)

# 4. Обучение модели
model = PLSA(n_topics=20, vocab_size=vocab_size, random_state=42)

history = model.fit(
    train_loader,
    n_epochs=100,
    val_loader=val_loader,
    patience=5,
    min_delta=10.0,
    verbose=True,
)

# 5. Анализ результатов
id_to_token = {v: k for k, v in token_to_id.items()}
vocab = [id_to_token.get(i, f"<unk_{i}>") for i in range(vocab_size)]

top_words = model.get_top_words(vocab, top_n=10)
for i, words in enumerate(top_words):
    print(f"Topic {i}: {', '.join(words)}")

# 6. Трансформация новых документов
# new_bow = ARTMCollator(vocab_size)([(0, [1, 2, 3, 1])])[1]
# theta = model.transform(new_bow)
