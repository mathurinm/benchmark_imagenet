from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import random
    import torch.utils.data
    import numpy as np
    import torchvision.transforms as transforms
    import os

    from torchvision.transforms import InterpolationMode
    from torchvision import datasets as datasets


FULL_SPLIT_TRAIN_VAL = 0.99
SPLIT_TRAIN_VAL = 0.9
TOTAL = 1331167
TOTAL_BLURRED = 1331063
SEED_FOR_SPLITTING = 0


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    # Name to select the dataset in the CLI and to display the results.
    name = "ImageNet"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        # "data": ["/datasets_local/ImageNet"], #[1281167],
        "blurred": [False],
        "random_augmentation_magnitude": [10],
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        trainset, valset, testset = get_imagenet_train_val_test(
            "/datasets_local/ImageNet",
            self.blurred,
            randAugLevel=self.random_augmentation_magnitude,
        )
        # TODO: remove hardcoded

        data_dict = dict(trainset=trainset, valset=valset, testset=testset)
        # The dictionary defines the keyword arguments for `Objective.set_data`
        return data_dict


def get_imagenet_train_val_test(
    imagenet_data, blurred, return_index=False, randAugLevel=None
):
    # Data loading code
    print(f"=> Getting data from {imagenet_data}")
    traindir = os.path.join(imagenet_data, "train_blurred" if blurred else "train")
    valdir = os.path.join(imagenet_data, "val_blurred" if blurred else "val")
    basic_transforms, augmentation_transforms = get_imagenet_transforms(randAugLevel)

    print(f"=> Creating datasets")

    print(
        f"Full dataset. Train set is splitted into {FULL_SPLIT_TRAIN_VAL} / {1 - FULL_SPLIT_TRAIN_VAL} for training and validation."
    )
    if not return_index:
        train_val_augmented = datasets.ImageFolder(traindir, augmentation_transforms)
    else:
        train_val_augmented = ReturnIndexDataset(traindir, augmentation_transforms)
    trainset, _, _, _, _, _ = imagetnet_dataset_splitted(
        SEED_FOR_SPLITTING,
        train_val_augmented,
        FULL_SPLIT_TRAIN_VAL,
        (1 - FULL_SPLIT_TRAIN_VAL),
        0,
        0,
        0,
        0,
    )

    if not return_index:
        train_val = datasets.ImageFolder(traindir, basic_transforms)
    else:
        train_val = ReturnIndexDataset(traindir, basic_transforms)
    _, valset, _, _, _, _ = imagetnet_dataset_splitted(
        SEED_FOR_SPLITTING,
        train_val,
        FULL_SPLIT_TRAIN_VAL,
        (1 - FULL_SPLIT_TRAIN_VAL),
        0,
        0,
        0,
        0,
    )

    if not return_index:
        testset = datasets.ImageFolder(valdir, basic_transforms)
    else:
        testset = ReturnIndexDataset(valdir, basic_transforms)

    return trainset, valset, testset


def get_imagenet_transforms(randAugLevel):
    """return basic and augmented transforms"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    basic_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    if randAugLevel is not None:
        augmentation_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, interpolation=InterpolationMode.BILINEAR
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(
                    interpolation=InterpolationMode.BILINEAR, magnitude=randAugLevel
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        augmentation_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, interpolation=InterpolationMode.BILINEAR
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return basic_transforms, augmentation_transforms


def imagetnet_dataset_splitted(
    seed, all_dataset, x_ttrain, x_tval, x_ttest, x_strain, x_sval, x_stest
):
    """Return: ttrain, tval, ttest, strain, sval, stest in this order"""
    label2inds = build_label_index(all_dataset.targets)
    list_label2inds_subset = labels2inds_split_subset(
        seed, label2inds, x_ttrain, x_tval, x_ttest, x_strain, x_sval, x_stest
    )
    return [
        subset_imagenet(label2inds_subset, all_dataset)
        for label2inds_subset in list_label2inds_subset
    ]


def build_label_index(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label2inds.get(label) is None:
            label2inds[label] = []
        label2inds[label].append(idx)
    return label2inds


def random_partition(seed, x, *sizes):
    assert len(x) == np.sum(sizes)
    random.Random(seed).shuffle(x)
    cumsum_sizes = np.cumsum(sizes)
    begin_idx = 0
    subsets = []
    for end_idx in cumsum_sizes:
        subset = x[begin_idx:end_idx]
        subsets.append(subset)
        begin_idx = end_idx
    return subsets


def split_class_indices(indices, x_ttrain, x_tval, x_ttest, x_strain, x_sval, x_stest):
    """x are proportions. The sum should be 1"""
    assert x_ttrain + x_tval + x_ttest + x_strain + x_sval + x_stest
    size = len(indices)
    size_ttrain = int(x_ttrain * size)
    size_tval = int(x_tval * size)
    size_ttest = int(x_ttest * size)
    size_strain = int(x_strain * size)
    size_sval = int(x_sval * size)
    size_stest = int(x_stest * size)

    subset_sum = (
        size_ttrain + size_tval + size_ttest + size_strain + size_sval + size_stest
    )
    remaining = size - subset_sum

    if remaining > 0:
        if remaining % 2 == 0:
            size_ttrain += remaining // 2
            size_strain += remaining // 2
        else:
            size_ttrain += remaining // 2 + 1
            size_strain += remaining // 2

    assert (
        size_ttrain + size_tval + size_ttest + size_strain + size_sval + size_stest
        == size
    )

    return size_ttrain, size_tval, size_ttest, size_strain, size_sval, size_stest


def labels2inds_split_subset(
    seed, label2inds, x_ttrain, x_tval, x_ttest, x_strain, x_sval, x_stest
):
    list_label2inds_subset = [{}, {}, {}, {}, {}, {}]
    for key in label2inds.keys():
        indices = label2inds[key]
        sizes = split_class_indices(
            indices, x_ttrain, x_tval, x_ttest, x_strain, x_sval, x_stest
        )
        subsets = random_partition(seed, indices, *sizes)
        for i, subset in enumerate(subsets):
            list_label2inds_subset[i][key] = subset

    return list_label2inds_subset


def subset_imagenet(label2inds_subset, all_dataset):
    assert len(label2inds_subset) == 1000
    all_indices = []
    for img_indices in label2inds_subset.values():
        all_indices += img_indices
    subset_dataset = torch.utils.data.Subset(all_dataset, all_indices)
    return subset_dataset


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, lab, idx
