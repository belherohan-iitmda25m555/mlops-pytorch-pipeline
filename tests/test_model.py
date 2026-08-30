"""Unit tests for the Fashion-MNIST CNN."""

import torch

from src.model import FashionCNN


def test_model_output_shape() -> None:
    """The model must return one score for each of 10 classes."""

    model = FashionCNN(num_classes=10)
    inputs = torch.randn(2, 1, 28, 28)

    outputs = model(inputs)

    assert outputs.shape == (2, 10)


def test_model_probabilities_sum_to_one() -> None:
    """Softmax probabilities must sum to one per image."""

    model = FashionCNN(num_classes=10)
    inputs = torch.randn(2, 1, 28, 28)

    outputs = model(inputs)
    probabilities = torch.softmax(outputs, dim=1)

    expected = torch.ones(2)

    assert torch.allclose(
        probabilities.sum(dim=1),
        expected,
        atol=1e-6,
    )