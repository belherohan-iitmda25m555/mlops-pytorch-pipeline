"""FastAPI application for Fashion-MNIST model inference."""

import io
import os
from contextlib import asynccontextmanager
from typing import Dict

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from src.model import FashionCNN


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "checkpoints/classifier_v1.pt",
)

model: FashionCNN | None = None

inference_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ]
)


def load_model(checkpoint_path: str) -> FashionCNN:
    """Load and return a trained Fashion-MNIST model."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    loaded_model = FashionCNN(
        num_classes=checkpoint.get("num_classes", 10)
    )

    loaded_model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    loaded_model.eval()

    return loaded_model


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load the checkpoint when the API starts."""

    del application

    global model

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    model = load_model(MODEL_PATH)

    yield

    model = None


app = FastAPI(
    title="Fashion-MNIST Classification API",
    description=(
        "Serve predictions from a trained PyTorch "
        "Fashion-MNIST classifier."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> Dict[str, str]:
    """Return HTTP 200 only when the model is loaded."""

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    return {
        "status": "healthy",
        "model_path": MODEL_PATH,
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
) -> Dict[str, object]:
    """Return predicted class and class probabilities."""

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    if image.content_type is None or not (
        image.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image",
        )

    try:
        image_bytes = await image.read()
        input_image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail="Unable to process the uploaded image",
        ) from error

    input_tensor = inference_transform(
        input_image
    ).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        probability_tensor = torch.softmax(
            logits,
            dim=1,
        )[0]

    predicted_index = int(
        probability_tensor.argmax().item()
    )

    probabilities = {
        class_name: round(float(probability), 6)
        for class_name, probability in zip(
            CLASS_NAMES,
            probability_tensor,
        )
    }

    return {
        "predicted_class": CLASS_NAMES[
            predicted_index
        ],
        "predicted_class_index": predicted_index,
        "probabilities": probabilities,
    }