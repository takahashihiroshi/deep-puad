import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
from torchvision import datasets, transforms


def make_toy_data(
    n_normal: int = 900,
    n_labeled_anomaly: int = 20,
    n_unlabeled_anomaly: int = 80,
    is_train=True,
    batch_size: int = 128,
) -> DataLoader:
    X_unlabeled_normal = np.zeros((n_normal, 2))
    for i, x in enumerate(np.linspace(-np.pi, np.pi, n_normal)):
        X_unlabeled_normal[i, 0] = x + np.random.normal(0, 0.1)
        X_unlabeled_normal[i, 1] = 3.0 * (np.sin(x) + np.random.normal(0, 0.2))

    X_unlabeled_anomaly = np.concatenate(
        [
            np.random.multivariate_normal(
                mean=[-np.pi / 2, np.pi / 2],
                cov=0.1 * np.eye(2),
                size=int(n_unlabeled_anomaly / 2),
            ),
            np.random.multivariate_normal(
                mean=[np.pi / 2, -np.pi / 2],
                cov=0.1 * np.eye(2),
                size=int(n_unlabeled_anomaly / 2),
            ),
        ]
    )

    X_anomaly = np.concatenate(
        [
            np.random.multivariate_normal(
                mean=[-np.pi / 2, np.pi / 2],
                cov=0.1 * np.eye(2),
                size=int(n_labeled_anomaly / 2),
            ),
            np.random.multivariate_normal(
                mean=[np.pi / 2, -np.pi / 2],
                cov=0.1 * np.eye(2),
                size=int(n_labeled_anomaly / 2),
            ),
        ]
    )

    X = np.concatenate([X_unlabeled_normal, X_unlabeled_anomaly, X_anomaly])
    if is_train:
        y = np.concatenate(
            [
                np.zeros(shape=n_normal),
                np.zeros(shape=n_unlabeled_anomaly),
                np.ones(shape=n_labeled_anomaly),
            ]
        )
    else:
        y = np.concatenate(
            [
                np.zeros(shape=n_normal),
                np.ones(shape=n_unlabeled_anomaly),
                np.ones(shape=n_labeled_anomaly),
            ]
        )

    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.int32))
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def load(
    name: str,
    batch_size: int = 128,
    normal_class: int = 0,
    unseen_anomaly: int = 9,
    n_train: int = 4500,
    n_valid: int = 500,
    n_test: int = 2000,
    n_unlabeled_normal: int = 4500,
    n_labeled_anomaly: int = 250,
    n_unlabeled_anomaly: int = 250,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    # dataset path
    path = f"datasets/{name}/" if name in ["CIFAR10", "SVHN"] else "datasets/"
    os.makedirs(path, exist_ok=True)

    # transform
    if name in ["CIFAR10", "SVHN"]:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((32, 32), antialias=True),
                transforms.Grayscale(),
                transforms.Lambda(lambda x: torch.flatten(x)),
            ]
        )
    else:  # MNIST, FashionMNIST
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((32, 32), antialias=True),
                transforms.Lambda(lambda x: torch.flatten(x)),
            ]
        )

    if name == "MNIST":
        train = datasets.MNIST(
            root=path, download=True, train=True, transform=transform
        )
        test = datasets.MNIST(
            root=path, download=True, train=False, transform=transform
        )
    elif name == "FashionMNIST":
        train = datasets.FashionMNIST(
            root=path, download=True, train=True, transform=transform
        )
        test = datasets.FashionMNIST(
            root=path, download=True, train=False, transform=transform
        )
    elif name == "CIFAR10":
        train = datasets.CIFAR10(
            root=path, download=True, train=True, transform=transform
        )
        test = datasets.CIFAR10(
            root=path, download=True, train=False, transform=transform
        )
    else:  # SVHN
        train = datasets.SVHN(
            root=path, download=True, split="train", transform=transform
        )
        test = datasets.SVHN(
            root=path, download=True, split="test", transform=transform
        )

    # Train
    train_indices = train.targets if name != "SVHN" else train.labels
    if not torch.is_tensor(train_indices):
        train_indices = torch.tensor(train_indices)

    train_normal_indices = (train_indices == normal_class).nonzero().squeeze().tolist()
    train_anomaly_indices = (
        torch.logical_and(
            train_indices != normal_class, train_indices != unseen_anomaly
        )
        .nonzero()
        .squeeze()
        .tolist()
    )

    train_normal_bag = random.sample(train_normal_indices, k=n_unlabeled_normal)
    train_anomaly_bag = random.sample(
        train_anomaly_indices, k=n_labeled_anomaly + n_unlabeled_anomaly
    )

    train_positive_bag = train_anomaly_bag[:n_labeled_anomaly]
    train_unlabeled_bag = train_normal_bag + train_anomaly_bag[n_labeled_anomaly:]

    for i in train_positive_bag:
        if name != "SVHN":
            train.targets[i] = 1
        else:
            train.labels[i] = 1

    for i in train_unlabeled_bag:
        if name != "SVHN":
            train.targets[i] = 0
        else:
            train.labels[i] = 0

    train_subset = Subset(train, train_positive_bag + train_unlabeled_bag)
    train_subset, valid_subset = random_split(train_subset, [n_train, n_valid])
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=True)

    # Test
    test_indices = test.targets if name != "SVHN" else test.labels
    if not torch.is_tensor(test_indices):
        test_indices = torch.tensor(test_indices)

    test_normal_indices = (test_indices == normal_class).nonzero().squeeze().tolist()
    test_anomaly_indices = (test_indices != normal_class).nonzero().squeeze().tolist()

    test_normal_bag = random.sample(
        test_normal_indices, k=min(len(test_normal_indices), 1000)
    )
    test_anomaly_bag = random.sample(
        test_anomaly_indices, k=n_test - len(test_normal_bag)
    )

    for i in test_anomaly_bag:
        if name != "SVHN":
            test.targets[i] = 1
        else:
            test.labels[i] = 1

    for i in test_normal_bag:
        if name != "SVHN":
            test.targets[i] = 0
        else:
            test.labels[i] = 0

    test_subset = Subset(test, test_anomaly_bag + test_normal_bag)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    for dataset_name in ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]:
        ret = load(dataset_name, batch_size=128)
        print(dataset_name)
        print("Train:", len(ret[0].dataset))  # type: ignore
        print("Valid:", len(ret[1].dataset))  # type: ignore
        print("Test:", len(ret[2].dataset))  # type: ignore
