import random


class MTDataLoader:
    """
    multi-class dataloader
    """

    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders
        self.total_samples = sum(map(lambda a: len(a), self.dataloaders))
        self.dataset = [None] * sum(len(dataloader.dataset) for dataloader in self.dataloaders)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        t = sum(([i] * len(a) for i, a in enumerate(self.dataloaders)), [])
        random.shuffle(t)

        iterators = [iter(a) for a in self.dataloaders]

        for x in t:
            yield next(iterators[x])


if __name__ == "__main__":

    class DummyIterator:
        def __init__(self, n, p) -> None:
            self.n = n
            self.p = p

        def __len__(self):
            return self.n

        def __iter__(self):
            return iter(range(self.p, self.p + self.n))

    big_loader = MTDataLoader(
        DummyIterator(10, 10),
        DummyIterator(10, 20),
        DummyIterator(10, 30),
    )

    print(", ".join(map(str, big_loader)))
    print(", ".join(map(str, big_loader)))
