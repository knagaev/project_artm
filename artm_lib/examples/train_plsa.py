from torch.utils.data import DataLoader

from artm_lib.data.collators import ARTMCollator
from artm_lib.data.dataset import ARTMDatasetParquet
from artm_lib.plsa.model import PLSA
from artm_lib.preprocessing.tokenizer import simple_tokenizer
from artm_lib.preprocessing.vocabulary import build_vocab_and_index_from_parquet

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
    dataset, batch_size=256, collate_fn=ARTMCollator(len(token_to_id)), num_workers=0
)

model = PLSA(n_topics=20, vocab_size=V, random_state=42)
model.fit(loader, n_epochs=10)

# Получить темы
top_words = model.get_top_words(vocab, top_n=10)
# После обучения модели
model = PLSA(n_topics=20, vocab_size=V)
model.fit(train_loader, n_epochs=10)

# Оценка на обучающей выборке
train_ppl = model.score_perplexity(train_loader)
print(f"Train Perplexity: {train_ppl:.2f}")

# Оценка на валидации (если есть)
val_ppl = model.score_perplexity(val_loader)
print(f"Validation Perplexity: {val_ppl:.2f}")
