import random
from itertools import chain
from typing import List, Optional

from utils.logger import logger


class MTDataLoader:
    """
    multi-class dataloader
    """

    def __init__(self, *dataloaders, weights: Optional[List[int]] = None):
        self.dataloaders = dataloaders
        self.weights = [1 for _ in dataloaders] if weights is None else weights
        self.total_samples = sum(len(ds) * w for ds, w in zip(self.dataloaders, self.weights))
        self.dataset = [None] * sum(len(dataloader.dataset) for dataloader in self.dataloaders)
        self.iterator_indices = sum(
            ([i] * len(a) * w for i, (a, w) in enumerate(zip(self.dataloaders, self.weights))), []
        )

        logger.info(
            f"Multi-task Dataloader loaded with weights: {':'.join(map(str, self.weights))}"
        )

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        random.shuffle(self.iterator_indices)

        iterators = [
            chain.from_iterable([iter(a) for _ in range(w)])
            for a, w in zip(self.dataloaders, self.weights)
        ]

        for x in self.iterator_indices:
            yield next(iterators[x])


if __name__ == "__main__":

    class DummyIterator:
        def __init__(self, n, p) -> None:
            self.n = n
            self.p = p
            self.dataset = list(range(self.p, self.p + self.n))

        def __len__(self):
            return self.n

        def __iter__(self):
            return iter(self.dataset)

    big_loader = MTDataLoader(
        DummyIterator(5, 10), DummyIterator(5, 20), DummyIterator(5, 30), weights=[2, 1, 1]
    )

    print(", ".join(map(str, big_loader)))
    print(", ".join(map(str, big_loader)))
