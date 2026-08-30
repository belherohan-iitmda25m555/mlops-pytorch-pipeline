"""Train the Fashion-MNIST classifier using YAML configuration."""

import json
import os
from pathlib import Path
from typing import Tuple

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from dataset import get_dataloaders
from model import FashionCNN


def log_metrics(**values: object) -> None:
    """Print structured JSON-line metrics to standard output."""
    print(json.dumps(values), flush=True)


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Return average loss and accuracy for a validation DataLoader."""

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            total_samples += labels.size(0)

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


def main() -> None:
    """Run model training and save the best checkpoint."""

    config_path = os.getenv(
        "CONFIG_PATH",
        "configs/training_config.yaml",
    )

    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    torch.manual_seed(config.get("seed", 42))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_loader, validation_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"].get("num_workers", 0),
    )

    model = FashionCNN(
        num_classes=config["model"]["num_classes"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )

    checkpoint_path = (
        Path(config["output"]["checkpoint_dir"])
        / config["output"]["model_name"]
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    patience = config["training"]["early_stopping_patience"]

    log_metrics(
        event="training_started",
        device=str(device),
        config_path=config_path,
    )

    for epoch in range(
        1,
        config["training"]["epochs"] + 1,
    ):
        model.train()

        total_training_loss = 0.0
        total_training_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_training_loss += (
                loss.item() * labels.size(0)
            )
            total_training_samples += labels.size(0)

        training_loss = (
            total_training_loss / total_training_samples
        )

        validation_loss, validation_accuracy = evaluate(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        log_metrics(
            event="epoch_completed",
            epoch=epoch,
            training_loss=round(training_loss, 6),
            validation_loss=round(validation_loss, 6),
            validation_accuracy=round(
                validation_accuracy,
                6,
            ),
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": config["model"]["num_classes"],
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                },
                checkpoint_path,
            )

            log_metrics(
                event="checkpoint_saved",
                path=str(checkpoint_path),
            )

        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                log_metrics(
                    event="early_stopping",
                    epoch=epoch,
                )
                break

    log_metrics(
        event="training_finished",
        best_validation_loss=round(
            best_validation_loss,
            6,
        ),
    )


if __name__ == "__main__":
    main()