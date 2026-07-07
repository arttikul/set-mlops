import json
import time
from pathlib import Path
from typing import List

import joblib
import torch
import torch.nn as nn
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel

MODEL_DIR = Path(__file__).resolve().parent / "model"

app = FastAPI()

# Метрики для Prometheus/Grafana (ДЗ-4): кількість передбачень за хвилину
# рахуємо як rate(emotion_predictions_total[1m])*60, середню latency -- як
# emotion_prediction_latency_seconds_sum / emotion_prediction_latency_seconds_count.
PREDICTIONS_TOTAL = Counter(
    "emotion_predictions_total", "Total number of texts classified", ["label"]
)
PREDICTION_LATENCY = Histogram(
    "emotion_prediction_latency_seconds", "Time spent processing an /invocations request"
)
app.mount("/metrics", make_asgi_app())


# Та сама архітектура, що й у training/train.py -- ваги завантажуються з
# артефакта в реєстрі, тож клас має лишатись ідентичним.
class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# Завантаження моделі, TF-IDF векторайзера та мапи класів
checkpoint = torch.load(MODEL_DIR / "model.pt", map_location="cpu")
vectorizer = joblib.load(MODEL_DIR / "vectorizer.pkl")
with open(MODEL_DIR / "labels.json") as f:
    classes = json.load(f)

model = EmotionClassifier(
    checkpoint["input_dim"],
    checkpoint["hidden_dim"],
    checkpoint["num_classes"],
    checkpoint["dropout"],
)
model.load_state_dict(checkpoint["state_dict"])
model.eval()


class TextsInput(BaseModel):
    texts: List[str]


@app.post("/invocations")
def predict(payload: TextsInput):
    start = time.perf_counter()

    features = vectorizer.transform(payload.texts).toarray()
    input_tensor = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1)

    labels = [classes[i] for i in probs.argmax(dim=1).tolist()]

    print("Input texts:", payload.texts)
    print("Predicted labels:", labels)

    PREDICTION_LATENCY.observe(time.perf_counter() - start)
    for label in labels:
        PREDICTIONS_TOTAL.labels(label=label).inc()

    return [
        {
            "text": text,
            "label": label,
            "probabilities": {cls: round(p, 4) for cls, p in zip(classes, row.tolist())},
        }
        for text, label, row in zip(payload.texts, labels, probs)
    ]


@app.get("/ping")
def ping():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
