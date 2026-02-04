# artm_lib/examples/train_plsa.py
import argparse
from pathlib import Path

from scipy.sparse import vstack
from sklearn.model_selection import train_test_split

from artm_lib.config import spacy_tokenizer as tokenizer
from artm_lib.data.collators import ARTMCollator
from artm_lib.data.dataset import ARTMDatasetParquet
from artm_lib.data.simple_loader import SimpleDataLoader
from artm_lib.plsa.model import PLSA
from artm_lib.preprocessing.format_utils import convert_csv_dir_to_parquet
from artm_lib.preprocessing.vocabulary import build_vocab_and_index_from_parquet


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(description="Train PLSA model")

    # Пути к данным
    parser.add_argument(
        "--csv_dir", type=str, default="./artm_lib/data/csvs", help="Directory with CSV files"
    )
    parser.add_argument(
        "--parquet_dir",
        type=str,
        default="./artm_lib/data/parquets",
        help="Directory for Parquet files",
    )
    parser.add_argument(
        "--text_column", type=str, default="Description", help="Text column name in CSV files"
    )

    # Параметры предобработки
    parser.add_argument(
        "--min_df", type=int, default=5, help="Minimum document frequency for vocabulary"
    )
    parser.add_argument(
        "--max_df",
        type=float,
        default=0.95,
        help="Maximum document frequency for vocabulary (0.0-1.0)",
    )

    # Параметры обучения
    parser.add_argument("--n_topics", type=int, default=20, help="Number of topics")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum number of EM iterations")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for data loading")

    # Воспроизводимость
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Флаги
    parser.add_argument(
        "--overwrite_parquet", action="store_true", help="Overwrite existing Parquet files"
    )

    parser.add_argument(
        "--val_split",
        type=float,
        default=0.0,
        help="Validation split fraction (0.0 = no validation)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Patience for early stopping (requires --val_split > 0)",
    )
    parser.add_argument(
        "--min_delta", type=float, default=1.0, help="Minimum delta for early stopping"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Создание директорий
    Path(args.parquet_dir).mkdir(parents=True, exist_ok=True)

    # 0. Конвертация CSV → Parquet
    print("Converting CSV to Parquet...")
    convert_csv_dir_to_parquet(
        csv_dir_str=args.csv_dir,
        parquet_dir_str=args.parquet_dir,
        text_column=args.text_column,
        overwrite=args.overwrite_parquet,
    )

    # 1. Построение словаря
    print("Building vocabulary...")
    token_to_id, doc_index = build_vocab_and_index_from_parquet(
        parquet_dir_str=args.parquet_dir,
        text_column=args.text_column,
        tokenizer=tokenizer,
        min_df=args.min_df,
        max_df=args.max_df,
    )
    V = len(token_to_id)
    print(f"Vocabulary size: {V}")

    # 2. Создание датасета и загрузчика
    dataset = ARTMDatasetParquet(
        doc_index=doc_index,
        token_to_id=token_to_id,
        text_column=args.text_column,
        tokenizer=tokenizer,
    )

    loader = SimpleDataLoader(
        dataset,
        collate_fn=ARTMCollator(V),
        batch_size=args.batch_size,
        shuffle=False,  # Для сбора полной матрицы шаффлинг не нужен
        seed=args.seed,
    )

    # 3. Сбор полной матрицы
    print("Building document-term matrix...")
    all_bows = []
    total_docs = len(dataset)
    processed_docs = 0

    for i, (_, bow) in enumerate(loader, 1):
        all_bows.append(bow)
        processed_docs += bow.shape[0]
        if i % 10 == 0 or processed_docs >= total_docs:
            print(f"Processed {processed_docs}/{total_docs} documents ({i} batches)")

    X = vstack(all_bows)
    print(f"Final matrix shape: {X.shape}")

    # Разделение (например, 90%/10%)
    if args.val_split > 0:
        X_train, X_val = train_test_split(X, test_size=args.val_split, random_state=args.seed)
        print(f"Train matrix: {X_train.shape}, Val matrix: {X_val.shape}")
    else:
        X_train, X_val = X, None

    # 4. Обучение модели
    print("Training PLSA model...")
    model = PLSA(n_topics=args.n_topics, vocab_size=V, random_state=args.seed)
    history = model.fit_full(
        X_train,
        max_iter=args.max_iter,
        tol=1e-4,
        val_X=X_val,
        patience=args.patience,
        min_delta=args.min_delta,
        verbose=True,
    )
    # 5. Результаты
    vocab = [""] * V
    for token, idx in token_to_id.items():
        vocab[idx] = token

    print("\nTop words per topic:")
    top_words = model.get_top_words(vocab, top_n=10)
    for i, words in enumerate(top_words):
        print(f"Topic {i}: {', '.join(words)}")

    perplexity = model.score_perplexity_from_matrix(X)
    print(f"\nFinal Perplexity: {perplexity:.2f}")

    # Сохранение истории (опционально)
    import json

    with open(f"training_history_seed_{args.seed}.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to training_history_seed_{args.seed}.json")


if __name__ == "__main__":
    main()
