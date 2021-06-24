from datasets.dataset import DSTDataset


class DSTDatasetForDSTForPrediction(DSTDataset):
    def __init__(self, *args, test_mode=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.test_mode = test_mode

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        if self.test_mode:
            return 10
        else:
            return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]
