# uv run python -m cartm.exps --models baseline --corpora bbc --sizes 100 --seeds 1

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from cartm.improved_model import AttentiveTopicModel
from .preprocessing import DatasetPreprocessor, BatchLoader
from .metrics.coherence import CoherenceMetric
from .metrics.perplexity import PerplexityMetric
from .metrics.phi_sparsity import SparsityMetric
from .metrics.topic_variance import TopicVarianceMetric
from .regularization.decorrelation import DecorrelationRegularization
from .regularization.sparsity import SparsityRegularization
#from .regularization.llmreg import LLMAlignmentRegularization
from .regularization.regularization_base import Regularization


Array = jax.Array

DEFAULT_CORPORA = ["bbc", "20ng_4cats", "agnews"]
DEFAULT_SIZES = ["100", "200", "500", "1000", "all"]

TWENTY_NG_CATEGORIES = [
    "comp.graphics",
    "rec.sport.hockey",
    "sci.space",
    "talk.politics.misc",
]


@dataclass
class CollectionInfo:
    name: str
    docs: list[str]
    labels: list[str] | None
    n_topics: int


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    use_llm_alignment: bool
    decor_tau: float = 0.0
    sparsity_tau: float = 0.0
    llm_reg_tau: float = 0.0
    llm_target_metric: str = "cosine"
    llm_temperature: float = 1.0


EXPERIMENT_LIBRARY: dict[str, ExperimentConfig] = {
    "baseline": ExperimentConfig(
        name="baseline",
        use_llm_alignment=False,
        decor_tau=0.0,
        sparsity_tau=0.0,
        llm_reg_tau=0.0,
    ),
    "baseline_decor": ExperimentConfig(
        name="baseline_decor",
        use_llm_alignment=False,
        decor_tau=1e-2,
        sparsity_tau=0.0,
        llm_reg_tau=0.0,
    ),
    "baseline_sparse": ExperimentConfig(
        name="baseline_sparse",
        use_llm_alignment=False,
        decor_tau=0.0,
        sparsity_tau=1e-3,
        llm_reg_tau=0.0,
    ),
    "llm_default": ExperimentConfig(
        name="llm_default",
        use_llm_alignment=True,
        decor_tau=0.0,
        sparsity_tau=0.0,
        llm_reg_tau=1e-3,
        llm_target_metric="cosine",
        llm_temperature=1.0,
    ),
    "llm_strong": ExperimentConfig(
        name="llm_strong",
        use_llm_alignment=True,
        decor_tau=0.0,
        sparsity_tau=0.0,
        llm_reg_tau=1e-2,
        llm_target_metric="cosine",
        llm_temperature=0.5,
    ),
}


@dataclass
class RunResult:
    collection: str
    corpus: str
    size_requested: str
    seed: int
    experiment_name: str
    model_name: str
    n_docs: int
    vocab_size: int
    n_topics: int
    coherence: float
    perplexity: float
    phi_sparsity: float
    topic_variance: float
    runtime_seconds: float

    # parameters used in the run; placed after main identifiers/metrics
    use_llm_alignment: bool
    ctx_len: int
    lr: float
    max_iter: int
    tol: float
    batch_size: int
    top_k: int
    decor_tau: float
    sparsity_tau: float
    llm_reg_tau: float
    llm_model_name: str
    llm_batch_size: int
    llm_target_metric: str
    llm_temperature: float


def sample_balanced(
    texts: Sequence[str],
    labels: Sequence[str],
    size: int | None,
    seed: int,
) -> tuple[list[str], list[str]]:
    texts = np.asarray(texts, dtype=object)
    labels = np.asarray(labels, dtype=object)

    if size is None or size >= len(texts):
        return texts.tolist(), labels.tolist()

    rng = np.random.default_rng(seed)
    unique_labels = sorted(set(labels.tolist()))
    per_label_base = size // len(unique_labels)
    remainder = size % len(unique_labels)

    selected_indices: list[int] = []
    for i, label in enumerate(unique_labels):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        take = min(len(idx), per_label_base + (1 if i < remainder else 0))
        selected_indices.extend(idx[:take].tolist())

    if len(selected_indices) < size:
        remaining = np.setdiff1d(
            np.arange(len(texts)),
            np.array(selected_indices, dtype=int),
            assume_unique=False,
        )
        rng.shuffle(remaining)
        need = size - len(selected_indices)
        selected_indices.extend(remaining[:need].tolist())

    rng.shuffle(selected_indices)
    return texts[selected_indices].tolist(), labels[selected_indices].tolist()


def load_bbc(size: int | None, seed: int) -> CollectionInfo:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install datasets: pip install datasets") from exc

    ds = load_dataset("SetFit/bbc-news")
    texts: list[str] = []
    labels: list[str] = []

    for split_name in ds.keys():
        split = ds[split_name]
        for row in split:
            text = row.get("text") or row.get("sentence") or row.get("content")
            label = row.get("label_text") or row.get("label_name") or row.get("label")
            if text is None:
                continue
            texts.append(str(text))
            labels.append(str(label))

    texts, labels = sample_balanced(texts, labels, size=size, seed=seed)
    return CollectionInfo(
        name=f"bbc_{size if size is not None else 'all'}",
        docs=texts,
        labels=labels,
        n_topics=5,
    )


def load_agnews(size: int | None, seed: int) -> CollectionInfo:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install datasets: pip install datasets") from exc

    ds = load_dataset("ag_news")
    texts: list[str] = []
    labels: list[str] = []

    label_names = None
    if "train" in ds and hasattr(ds["train"].features.get("label"), "names"):
        label_names = ds["train"].features["label"].names

    for split_name in ds.keys():
        split = ds[split_name]
        for row in split:
            text = row.get("text") or row.get("description") or row.get("content")
            label = row.get("label")
            if text is None:
                continue
            texts.append(str(text))
            if label_names is not None and isinstance(label, (int, np.integer)):
                labels.append(str(label_names[int(label)]))
            else:
                labels.append(str(label))

    texts, labels = sample_balanced(texts, labels, size=size, seed=seed)
    return CollectionInfo(
        name=f"agnews_{size if size is not None else 'all'}",
        docs=texts,
        labels=labels,
        n_topics=4,
    )


def load_20ng_4cats(size: int | None, seed: int, data_home: Path | None) -> CollectionInfo:
    try:
        from sklearn.datasets import fetch_20newsgroups
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install scikit-learn: pip install scikit-learn") from exc

    ds = fetch_20newsgroups(
        subset="all",
        categories=TWENTY_NG_CATEGORIES,
        shuffle=True,
        random_state=seed,
        remove=("headers", "footers", "quotes"),
        data_home=str(data_home) if data_home is not None else None,
    )

    texts = [str(x) for x in ds.data]
    labels = [str(ds.target_names[i]) for i in ds.target]
    texts, labels = sample_balanced(texts, labels, size=size, seed=seed)

    return CollectionInfo(
        name=f"20ng_4cats_{size if size is not None else 'all'}",
        docs=texts,
        labels=labels,
        n_topics=4,
    )


def load_collection(
    corpus: str,
    *,
    size: int | None,
    seed: int,
    data_home: Path | None,
) -> CollectionInfo:
    corpus = corpus.strip().lower()

    if corpus == "bbc":
        return load_bbc(size=size, seed=seed)
    if corpus == "agnews":
        return load_agnews(size=size, seed=seed)
    if corpus == "20ng_4cats":
        return load_20ng_4cats(size=size, seed=seed, data_home=data_home)

    raise ValueError(f"Unsupported corpus: {corpus}")


def build_bow_matrix_numpy(data, doc_bounds, vocab_size):
    data = np.asarray(data)
    doc_bounds = np.asarray(doc_bounds)

    n_docs = int(doc_bounds.shape[0] - 1)
    bow = np.zeros((n_docs, vocab_size), dtype=np.int32)

    for d in range(n_docs):
        start = int(doc_bounds[d])
        end = int(doc_bounds[d + 1])
        tokens = data[start:end]
        if len(tokens) > 0:
            np.add.at(bow[d], tokens, 1)

    return bow


def ordered_terms(vocabulary: dict[str, int]) -> list[str]:
    terms = [""] * len(vocabulary)
    for term, idx in vocabulary.items():
        terms[idx] = term
    return terms


def encode_vocabulary_with_llm(
    vocabulary: dict[str, int],
    *,
    model_name: str,
    batch_size: int,
    cache_dir: Path | None,
) -> Array:
    terms = ordered_terms(vocabulary)
    cache_path = None

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        signature = str(abs(hash(tuple(terms))))
        cache_path = cache_dir / f"{model_name.replace('/', '__')}__{signature}.npy"
        if cache_path.exists():
            return jnp.asarray(np.load(cache_path), dtype=jnp.float32)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install sentence-transformers: pip install sentence-transformers") from exc

    encoder = SentenceTransformer(model_name)
    embeddings = encoder.encode(
        terms,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if cache_path is not None:
        np.save(cache_path, embeddings)

    return jnp.asarray(embeddings, dtype=jnp.float32)


def build_llm_topic_targets(
    *,
    word_embeddings: Array,
    topic_centers: Array,
    tau: float = 1.0,
    eps: float = 1e-12,
) -> Array:
    tau = jnp.maximum(jnp.asarray(tau, dtype=word_embeddings.dtype), eps)
    d2_tw = jnp.sum(
        (topic_centers[:, None, :] - word_embeddings[None, :, :]) ** 2,
        axis=-1,
    )
    logits_tw = -d2_tw / tau
    q_tw = jax.nn.softmax(logits_tw, axis=0)
    return q_tw


def build_llm_topic_targets_cosine(
    *,
    word_embeddings: Array,
    topic_centers: Array,
    tau: float = 1.0,
    eps: float = 1e-12,
) -> Array:
    tau = jnp.maximum(jnp.asarray(tau, dtype=word_embeddings.dtype), eps)

    we = word_embeddings / (jnp.linalg.norm(word_embeddings, axis=1, keepdims=True) + eps)
    tc = topic_centers / (jnp.linalg.norm(topic_centers, axis=1, keepdims=True) + eps)

    sim_tw = tc @ we.T
    logits_tw = sim_tw / tau
    q_tw = jax.nn.softmax(logits_tw, axis=0)
    return q_tw


def build_q_tw(
    *,
    word_embeddings: Array,
    n_topics: int,
    metric: str,
    temperature: float,
    seed: int,
) -> Array:
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install scikit-learn: pip install scikit-learn") from exc

    centers = KMeans(
        n_clusters=n_topics,
        random_state=seed,
        n_init=10,
    ).fit(np.asarray(word_embeddings)).cluster_centers_
    centers = jnp.asarray(centers, dtype=jnp.float32)

    if metric == "cosine":
        return build_llm_topic_targets_cosine(
            word_embeddings=word_embeddings,
            topic_centers=centers,
            tau=temperature,
        )
    if metric == "euclidean":
        return build_llm_topic_targets(
            word_embeddings=word_embeddings,
            topic_centers=centers,
            tau=temperature,
        )

    raise ValueError(f"Unknown metric: {metric}")


def make_regularizations(
    *,
    vocab_size: int,
    n_topics: int,
    decor_tau: float,
    sparsity_tau: float,
    q_tw: Array | None,
    llm_reg_tau: float,
) -> list[Regularization]:
    regs: list[Regularization] = []

    if decor_tau > 0:
        regs.append(DecorrelationRegularization(tau=decor_tau))

    if sparsity_tau > 0:
        alpha = jnp.ones((vocab_size, n_topics), dtype=jnp.float32)
        regs.append(SparsityRegularization(alpha=alpha, tau=sparsity_tau))

    if q_tw is not None and llm_reg_tau > 0:
        regs.append(LLMAlignmentRegularization(q_tw=q_tw, tau=llm_reg_tau))

    return regs


def instantiate_model(
    *,
    vocab_size: int,
    ctx_len: int,
    n_topics: int,
    regularizers: list[Regularization],
) -> Any:
    return AttentiveTopicModel(
        vocab_size=vocab_size,
        ctx_len=ctx_len,
        n_topics=n_topics,
        regularizers=regularizers,
    )


def zero_grad_like(x: Array) -> Array:
    return jnp.zeros_like(x)


def evaluate_final_state(model: Any, *, data: Array, doc_bounds: Array) -> tuple[Array, Array, Array]:
    phi_it, _, theta, _ = model._step(
        batch=data,
        ctx_bounds=doc_bounds,
        phi=model.phi,
        n_t=model.n_t,
        grad_reg=zero_grad_like,
    )
    return phi_it, model.phi, theta


def make_metrics(bow: Array, top_k: int, perplexity_chunk_size: int = 10000) -> dict[str, Any]:
    return {
        "coherence": CoherenceMetric(data=bow, top_k=top_k),
        "perplexity": PerplexityMetric(chunk_size=perplexity_chunk_size),
        "phi_sparsity": SparsityMetric(),
        "topic_variance": TopicVarianceMetric(distance_metric="jaccard", top_k=top_k),
    }


def resolve_experiments(args: argparse.Namespace) -> list[ExperimentConfig]:
    if args.experiments is not None:
        unknown = [name for name in args.experiments if name not in EXPERIMENT_LIBRARY]
        if unknown:
            raise ValueError(
                f"Unknown experiments: {unknown}. "
                f"Available: {sorted(EXPERIMENT_LIBRARY.keys())}"
            )
        return [EXPERIMENT_LIBRARY[name] for name in args.experiments]

    if args.models is not None:
        experiments: list[ExperimentConfig] = []
        for model_kind in args.models:
            if model_kind == "baseline":
                experiments.append(
                    ExperimentConfig(
                        name="baseline_cli",
                        use_llm_alignment=False,
                        decor_tau=args.decor_tau,
                        sparsity_tau=args.sparsity_tau,
                        llm_reg_tau=0.0,
                        llm_target_metric=args.llm_target_metric,
                        llm_temperature=args.llm_temperature,
                    )
                )
            elif model_kind == "llm":
                experiments.append(
                    ExperimentConfig(
                        name="llm_cli",
                        use_llm_alignment=True,
                        decor_tau=args.decor_tau,
                        sparsity_tau=args.sparsity_tau,
                        llm_reg_tau=args.llm_reg_tau,
                        llm_target_metric=args.llm_target_metric,
                        llm_temperature=args.llm_temperature,
                    )
                )
        return experiments

    return [
        EXPERIMENT_LIBRARY["baseline"],
        EXPERIMENT_LIBRARY["llm_default"],
    ]


def parse_size_spec(size_spec: str) -> int | None:
    size_spec = str(size_spec).strip().lower()
    if size_spec == "all":
        return None

    size = int(size_spec)
    if size <= 0:
        raise ValueError(f"Size must be positive, got {size}")
    return size


def run_one_model(
    *,
    experiment_name: str,
    model_name: str,
    corpus: str,
    size_requested: str,
    docs: Sequence[str],
    n_topics: int,
    ctx_len: int,
    lr: float,
    max_iter: int,
    tol: float,
    seed: int,
    batch_size: int,
    top_k: int,
    decor_tau: float,
    sparsity_tau: float,
    llm_model_name: str,
    llm_batch_size: int,
    llm_cache_dir: Path | None,
    llm_target_metric: str,
    llm_temperature: float,
    llm_reg_tau: float,
    use_llm_alignment: bool,
) -> RunResult:
    preprocessor = DatasetPreprocessor()
    data, doc_bounds = preprocessor.fit_transform(docs, return_doc_bounds=True)
    vocab = preprocessor.vocabulary
    vocab_size = len(vocab)

    if vocab_size < n_topics:
        raise ValueError(
            f"Vocabulary size {vocab_size} is smaller than n_topics={n_topics} for {model_name}."
        )

    q_tw = None
    if use_llm_alignment:
        word_embeddings = encode_vocabulary_with_llm(
            vocabulary=vocab,
            model_name=llm_model_name,
            batch_size=llm_batch_size,
            cache_dir=llm_cache_dir,
        )
        q_tw = build_q_tw(
            word_embeddings=word_embeddings,
            n_topics=n_topics,
            metric=llm_target_metric,
            temperature=llm_temperature,
            seed=seed,
        )

    regularizers = make_regularizations(
        vocab_size=vocab_size,
        n_topics=n_topics,
        decor_tau=decor_tau,
        sparsity_tau=sparsity_tau,
        q_tw=q_tw,
        llm_reg_tau=llm_reg_tau,
    )

    model = instantiate_model(
        vocab_size=vocab_size,
        ctx_len=ctx_len,
        n_topics=n_topics,
        regularizers=regularizers,
    )

    loader = BatchLoader(data, doc_bounds, batch_size=batch_size)

    start_time = time.perf_counter()

    model.fit(
        data=loader,
        lr=lr,
        max_iter=max_iter,
        tol=tol,
        verbose=0,
        seed=seed,
    )

    phi_it, phi_wt, theta = evaluate_final_state(model, data=data, doc_bounds=doc_bounds)

    phi_np = np.asarray(model.phi)
    print("phi shape:", phi_np.shape)
    print("phi first row:", phi_np[0])
    print("phi first column:", phi_np[:, 0])

    bow = build_bow_matrix_numpy(data, doc_bounds, vocab_size)
    metrics = make_metrics(
        bow=jnp.asarray(bow),
        top_k=top_k,
        perplexity_chunk_size=10000,
    )

    scores: dict[str, float] = {}

    print("computing coherence")
    coh = metrics["coherence"](phi_it=phi_it, phi_wt=phi_wt, theta=theta)
    scores["coherence"] = float(jax.device_get(coh))

    print("computing perplexity")
    ppl = metrics["perplexity"](phi_it=phi_it, phi_wt=phi_wt, theta=theta)
    scores["perplexity"] = float(jax.device_get(ppl))

    print("computing sparsity")
    phi_sp = metrics["phi_sparsity"](phi_it=phi_it, phi_wt=phi_wt, theta=theta)
    scores["phi_sparsity"] = float(jax.device_get(phi_sp))

    print("computing topic_variance")
    tv = metrics["topic_variance"](phi_it=phi_it, phi_wt=phi_wt, theta=theta)
    scores["topic_variance"] = float(jax.device_get(tv))

    jax.block_until_ready(tv)
    runtime_seconds = time.perf_counter() - start_time

    return RunResult(
        collection="",
        corpus=corpus,
        size_requested=size_requested,
        seed=seed,
        experiment_name=experiment_name,
        model_name=model_name,
        n_docs=len(docs),
        vocab_size=vocab_size,
        n_topics=n_topics,
        coherence=scores["coherence"],
        perplexity=scores["perplexity"],
        phi_sparsity=scores["phi_sparsity"],
        topic_variance=scores["topic_variance"],
        runtime_seconds=runtime_seconds,
        use_llm_alignment=use_llm_alignment,
        ctx_len=ctx_len,
        lr=lr,
        max_iter=max_iter,
        tol=tol,
        batch_size=batch_size,
        top_k=top_k,
        decor_tau=decor_tau,
        sparsity_tau=sparsity_tau,
        llm_reg_tau=llm_reg_tau,
        llm_model_name=llm_model_name,
        llm_batch_size=llm_batch_size,
        llm_target_metric=llm_target_metric,
        llm_temperature=llm_temperature,
    )


def summarize(results: list[RunResult]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[RunResult]] = {}
    for row in results:
        key = (
            row.collection,
            row.corpus,
            row.size_requested,
            row.experiment_name,
            row.model_name,
            row.use_llm_alignment,
            row.ctx_len,
            row.lr,
            row.max_iter,
            row.tol,
            row.batch_size,
            row.top_k,
            row.decor_tau,
            row.sparsity_tau,
            row.llm_reg_tau,
            row.llm_model_name,
            row.llm_batch_size,
            row.llm_target_metric,
            row.llm_temperature,
        )
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items(), key=lambda x: x[0]):
        (
            collection,
            corpus,
            size_requested,
            experiment_name,
            model_name,
            use_llm_alignment,
            ctx_len,
            lr,
            max_iter,
            tol,
            batch_size,
            top_k,
            decor_tau,
            sparsity_tau,
            llm_reg_tau,
            llm_model_name,
            llm_batch_size,
            llm_target_metric,
            llm_temperature,
        ) = key

        summary.append(
            {
                "collection": collection,
                "corpus": corpus,
                "size_requested": size_requested,
                "experiment_name": experiment_name,
                "model_name": model_name,
                "n_runs": len(rows),
                "n_docs": int(np.mean([r.n_docs for r in rows])),
                "n_topics": int(np.mean([r.n_topics for r in rows])),
                "vocab_size_mean": float(np.mean([r.vocab_size for r in rows])),
                "coherence_mean": float(np.mean([r.coherence for r in rows])),
                "coherence_std": float(np.std([r.coherence for r in rows])),
                "perplexity_mean": float(np.mean([r.perplexity for r in rows])),
                "perplexity_std": float(np.std([r.perplexity for r in rows])),
                "phi_sparsity_mean": float(np.mean([r.phi_sparsity for r in rows])),
                "phi_sparsity_std": float(np.std([r.phi_sparsity for r in rows])),
                "topic_variance_mean": float(np.mean([r.topic_variance for r in rows])),
                "topic_variance_std": float(np.std([r.topic_variance for r in rows])),
                "runtime_seconds_mean": float(np.mean([r.runtime_seconds for r in rows])),
                "runtime_seconds_std": float(np.std([r.runtime_seconds for r in rows])),
                "use_llm_alignment": use_llm_alignment,
                "ctx_len": ctx_len,
                "lr": lr,
                "max_iter": max_iter,
                "tol": tol,
                "batch_size": batch_size,
                "top_k": top_k,
                "decor_tau": decor_tau,
                "sparsity_tau": sparsity_tau,
                "llm_reg_tau": llm_reg_tau,
                "llm_model_name": llm_model_name,
                "llm_batch_size": llm_batch_size,
                "llm_target_metric": llm_target_metric,
                "llm_temperature": llm_temperature,
            }
        )
    return summary


def save_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selected topic-model experiments on selected corpora."
    )

    parser.add_argument(
        "--corpora",
        nargs="+",
        choices=["bbc", "20ng_4cats", "agnews"],
        default=DEFAULT_CORPORA,
        help="Corpora to run, e.g. bbc 20ng_4cats agnews",
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        default=DEFAULT_SIZES,
        help="Numbers of documents to sample from each corpus, e.g. 10 100 500 all",
    )

    parser.add_argument("--output-dir", type=Path, default=Path("./experiment_outputs"))
    parser.add_argument("--data-home", type=Path, default=Path("./dataset_cache"))

    parser.add_argument("--ctx-len", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    parser.add_argument("--decor-tau", type=float, default=0.0)
    parser.add_argument("--sparsity-tau", type=float, default=0.0)

    parser.add_argument("--llm-model-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--llm-batch-size", type=int, default=128)
    parser.add_argument("--llm-cache-dir", type=Path, default=Path("./llm_vocab_cache"))
    parser.add_argument("--llm-target-metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--llm-temperature", type=float, default=1.0)
    parser.add_argument("--llm-reg-tau", type=float, default=1e-3)

    parser.add_argument(
        "--n-topics-override",
        type=int,
        default=None,
        help="If set, use this number of topics for all collections instead of dataset default.",
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help=f"Named experiments to run. Available: {', '.join(sorted(EXPERIMENT_LIBRARY.keys()))}",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["baseline", "llm"],
        default=None,
        help="Simple direct selection of models. Ignored if --experiments is provided.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_home.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    experiments = resolve_experiments(args)
    all_results: list[RunResult] = []

    print("Selected experiments:")
    for exp in experiments:
        print(
            f"  - {exp.name}: use_llm_alignment={exp.use_llm_alignment}, "
            f"decor_tau={exp.decor_tau}, sparsity_tau={exp.sparsity_tau}, "
            f"llm_reg_tau={exp.llm_reg_tau}, llm_metric={exp.llm_target_metric}, "
            f"llm_temp={exp.llm_temperature}"
        )

    for corpus in args.corpora:
        for size_spec in args.sizes:
            size = parse_size_spec(size_spec)
            collection_label = f"{corpus}_{size if size is not None else 'all'}"
            print(f"\n=== Collection: {collection_label} ===")

            for seed in args.seeds:
                collection = load_collection(
                    corpus,
                    size=size,
                    seed=seed,
                    data_home=args.data_home,
                )
                n_topics = args.n_topics_override or collection.n_topics
                print(f"seed={seed} docs={len(collection.docs)} topics={n_topics}")

                for exp in experiments:
                    model_name = "llm_aligned" if exp.use_llm_alignment else "baseline"
                    print(f"running experiment={exp.name} model={model_name}")

                    result = run_one_model(
                        experiment_name=exp.name,
                        model_name=model_name,
                        corpus=corpus,
                        size_requested=str(size_spec),
                        docs=collection.docs,
                        n_topics=n_topics,
                        ctx_len=args.ctx_len,
                        lr=args.lr,
                        max_iter=args.max_iter,
                        tol=args.tol,
                        seed=seed,
                        batch_size=args.batch_size,
                        top_k=args.top_k,
                        decor_tau=exp.decor_tau,
                        sparsity_tau=exp.sparsity_tau,
                        llm_model_name=args.llm_model_name,
                        llm_batch_size=args.llm_batch_size,
                        llm_cache_dir=args.llm_cache_dir,
                        llm_target_metric=exp.llm_target_metric,
                        llm_temperature=exp.llm_temperature,
                        llm_reg_tau=exp.llm_reg_tau,
                        use_llm_alignment=exp.use_llm_alignment,
                    )
                    result.collection = collection.name
                    all_results.append(result)

                    print(
                        f"[{exp.name}] seed={seed} coh={result.coherence:.4f} "
                        f"ppl={result.perplexity:.4f} phi_sp={result.phi_sparsity:.4f} "
                        f"tv={result.topic_variance:.4f} time={result.runtime_seconds:.2f}s"
                    )

    raw_rows = [asdict(x) for x in all_results]
    summary_rows = summarize(all_results)

    save_csv(args.output_dir / "raw_results.csv", raw_rows)
    save_csv(args.output_dir / "summary_results.csv", summary_rows)

    print("\nSaved:")
    print(args.output_dir / "raw_results.csv")
    print(args.output_dir / "summary_results.csv")


if __name__ == "__main__":
    main()