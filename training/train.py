import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR.parent / "dataset" / "emotion-dataset-labeled.csv"
ARTIFACT_DIR = SCRIPT_DIR / "artifacts"


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a text emotion classifier (TF-IDF + MLP) with W&B tracking"
    )
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--max-features", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


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


def iterate_batches(X, y, batch_size, shuffle):
    n = X.shape[0]
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        batch_idx = idx[start:start + batch_size]
        # Densify only the current mini-batch -- the full TF-IDF matrix is too
        # large to fit in memory as a dense array.
        xb = torch.tensor(X[batch_idx].toarray(), dtype=torch.float32)
        yb = torch.tensor(y[batch_idx], dtype=torch.long)
        yield xb, yb


def run_epoch(model, X, y, batch_size, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in iterate_batches(X, y, batch_size, shuffle=training):
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(xb)
            loss = criterion(logits, yb)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(yb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        total += len(yb)
    return total_loss / total, correct / total


def main():
    args = parse_args()

    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        print("Warning: WANDB_API_KEY not found in environment variables")
    entity = os.getenv("WANDB_ENTITY") or None

    print(f"Loading dataset from {DATASET_PATH} ...")
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=["text", "label"])

    classes = sorted(df["label"].unique())
    label_to_idx = {label: i for i, label in enumerate(classes)}
    y_all = df["label"].map(label_to_idx).to_numpy()

    X_train_text, X_val_text, y_train, y_val = train_test_split(
        df["text"].to_numpy(), y_all, test_size=0.1, stratify=y_all, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    # max_features may exceed the actual vocabulary size found by the vectorizer.
    input_dim = len(vectorizer.vocabulary_)

    run = wandb.init(
        project="emotion-classification",
        entity=entity,
        name=args.run_name,
        config={
            **vars(args),
            "input_dim": input_dim,
            "num_classes": len(classes),
            "train_size": X_train.shape[0],
            "val_size": X_val.shape[0],
        },
    )

    model = EmotionClassifier(input_dim, args.hidden_dim, len(classes), args.dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    wandb.watch(model, log_freq=100)

    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(
            model, X_train, y_train, args.batch_size, criterion, optimizer
        )
        val_loss, val_acc = run_epoch(model, X_val, y_val, args.batch_size, criterion)
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"epoch {epoch + 1}/{args.epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    # Final per-class report on the validation set
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in iterate_batches(X_val, y_val, args.batch_size, shuffle=False):
            preds.append(model(xb).argmax(dim=1).numpy())
    y_pred = np.concatenate(preds)
    report_text = classification_report(y_val, y_pred, target_names=classes)
    print(report_text)
    report_df = pd.DataFrame(
        classification_report(y_val, y_pred, target_names=classes, output_dict=True)
    ).transpose().reset_index()
    wandb.log({"val_classification_report": wandb.Table(dataframe=report_df)})

    # Save model + preprocessing artifacts
    ARTIFACT_DIR.mkdir(exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "num_classes": len(classes),
        },
        ARTIFACT_DIR / "model.pt",
    )
    joblib.dump(vectorizer, ARTIFACT_DIR / "vectorizer.pkl")
    with open(ARTIFACT_DIR / "labels.json", "w") as f:
        json.dump(classes, f)

    artifact = wandb.Artifact("emotion-classifier", type="model")
    artifact.add_dir(str(ARTIFACT_DIR))
    logged_artifact = run.log_artifact(artifact)
    logged_artifact.wait()

    # New-style W&B Registry: collections live under "wandb-registry-<registry>",
    # scoped by the run's entity/team rather than an entity prefix in the path.
    target_path = "wandb-registry-model/emotion-classifier"
    try:
        run.link_artifact(artifact=logged_artifact, target_path=target_path)
        print(f"Linked artifact to model registry: {target_path}")
    except Exception as e:
        print(f"Warning: failed to link artifact to model registry ({target_path}): {e}")

    print(f"W&B run: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
