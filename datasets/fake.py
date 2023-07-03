from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Fake dataset"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        "trainsize": [1281167],
        "valsize": [50000],
        "testsize": [50000],
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        data_dict = dict(
            trainset = datasets.FakeData(
            self.trainsize, (3, 224, 224), 1000, transforms.ToTensor()),
            valset = datasets.FakeData(
                self.valsize, (3, 224, 224), 1000, transforms.ToTensor()),
            testset = datasets.FakeData(
                self.testsize, (3, 224, 224), 1000, transforms.ToTensor())
        )
        # The dictionary defines the keyword arguments for `Objective.set_data`
        return data_dict
