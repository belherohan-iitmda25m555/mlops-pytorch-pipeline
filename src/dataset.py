"""Fashion-MNIST dataset and DataLoader utilities."""

from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_transforms() -> transforms.Compose:
    """Return preprocessing transforms for Fashion-MNIST images."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )


def get_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Download Fashion-MNIST and return training and validation loaders."""

    dataset = datasets.FashionMNIST(
        root=Path(data_dir),
        train=True,
        download=True,
        transform=get_transforms(),
    )

    train_dataset, validation_dataset = random_split(
        dataset,
        [55_000, 5_000],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, validation_loader