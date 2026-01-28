# artm_lib/data/simple_loader.py
"""
Minimal DataLoader replacement without PyTorch dependency.
Compatible with ARTMDatasetParquet and ARTMCollator.
"""

import random
from typing import Any, Callable, Optional, Iterator

from scipy.sparse import csr_matrix


class SimpleDataLoader:
    """
    A lightweight DataLoader alternative that does not require PyTorch.

    Args:
        dataset: Any object with __len__ and __getitem__ methods.
        collate_fn: Callable that takes a list of dataset items and returns a batch.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data before each epoch.
        seed: Random seed for shuffling (optional).
    """

    def __init__(
        self,
        dataset: Any,
        collate_fn: Callable[[list[Any]], tuple[list[int], csr_matrix]],
        batch_size: int = 32,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.dataset = dataset
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        # Create initial indices
        self.indices = list(range(len(dataset)))

    def __iter__(self) -> Iterator[tuple[list[int], csr_matrix]]:
        """Yield batches of data."""
        # Shuffle indices if needed
        if self.shuffle:
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(self.indices)

        batch = []
        for idx in self.indices:
            item = self.dataset[idx]
            batch.append(item)

            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []

        # Yield last partial batch if exists
        if batch:
            yield self.collate_fn(batch)

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
