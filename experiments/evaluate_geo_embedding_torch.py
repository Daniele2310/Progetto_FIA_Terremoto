"""
Esperimento completo PyTorch per valutare il contributo dei geo_level_id.

Confronta sullo stesso split stratificato:
    1. mlp_no_geo:
       rete MLP sulle feature preprocessate, escludendo geo_level_1/2/3_id.
    2. geo_embedding_only:
       rete che usa solo embedding dei tre geo_level_id.
    3. mlp_with_geo_embeddings:
       rete MLP sulle feature preprocessate + embedding dei tre geo_level_id.

La metrica primaria e' F1-micro, equivalente all'accuracy nel caso
single-label multiclass, ma mantenuta per coerenza con DrivenData.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.data_selection import get_balanced_sample, get_stratified_sample


TARGET_COL = "damage_grade"
ID_COL = "building_id"
GEO_COLS = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
RANDOM_STATE = 42


@dataclass(frozen=True)
class SplitData:
    x_dense_train: np.ndarray
    x_dense_val: np.ndarray
    x_geo_train: np.ndarray
    x_geo_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    dense_cols: list[str]
    label_to_idx: dict[int, int]
    idx_to_label: dict[int, int]
    n_geo_categories: list[int]


class TabularGeoNet(nn.Module):
    def __init__(
        self,
        n_dense_features: int,
        n_geo_categories: list[int],
        embedding_dims: list[int],
        hidden_dims: list[int],
        dropout: float,
        n_classes: int,
        use_dense: bool = True,
        use_geo: bool = True,
    ):
        super().__init__()
        self.use_dense = use_dense
        self.use_geo = use_geo

        if use_geo:
            self.embeddings = nn.ModuleList(
                [
                    nn.Embedding(num_embeddings=n_categories + 1, embedding_dim=dim, padding_idx=0)
                    for n_categories, dim in zip(n_geo_categories, embedding_dims)
                ]
            )
            geo_dim = int(sum(embedding_dims))
        else:
            self.embeddings = nn.ModuleList()
            geo_dim = 0

        input_dim = (n_dense_features if use_dense else 0) + geo_dim
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x_dense: torch.Tensor, x_geo: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.use_dense:
            parts.append(x_dense)
        if self.use_geo:
            geo_embeddings = [
                embedding(x_geo[:, idx])
                for idx, embedding in enumerate(self.embeddings)
            ]
            parts.append(torch.cat(geo_embeddings, dim=1))

        x = torch.cat(parts, dim=1)
        return self.network(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch delle geo_id con embedding neurali."
    )
    parser.add_argument("--sample-mode", choices=["full", "balanced", "stratified"], default="full")
    parser.add_argument("--max-per-class", type=int, default=20000)
    parser.add_argument("--n-samples", type=int, default=60000)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--embedding-dims", type=str, default="4,16,32")
    parser.add_argument("--hidden-dims", type=str, default="128,64")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["no_geo", "geo_only", "with_geo"],
        default=["no_geo", "geo_only", "with_geo"],
        help="Subset di modelli da eseguire.",
    )
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="Pesi classe nella CrossEntropyLoss. F1-micro spesso preferisce none.",
    )
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_int_list(raw: str, expected_len: int | None = None) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"Valori attesi: {expected_len}, ricevuti: {values}")
    return values


def load_dataset() -> pd.DataFrame:
    path = PROJECT_ROOT / "Data" / "preprocessed" / "train_features_labels_preprocessed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset preprocessato non trovato: {path}. Esegui prima main.py.")

    df = pd.read_csv(path)
    missing = [col for col in [*GEO_COLS, TARGET_COL] if col not in df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti nel dataset: {missing}")
    return df


def sample_dataset(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.sample_mode == "balanced":
        return get_balanced_sample(df, TARGET_COL, args.max_per_class, args.random_state)
    if args.sample_mode == "stratified":
        return get_stratified_sample(df, TARGET_COL, args.n_samples, args.random_state)
    return df.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)


def fit_geo_mappings(train_df: pd.DataFrame) -> tuple[dict[str, dict[int, int]], list[int]]:
    mappings = {}
    n_categories = []
    for col in GEO_COLS:
        values = sorted(train_df[col].astype(int).unique().tolist())
        mappings[col] = {value: idx + 1 for idx, value in enumerate(values)}
        n_categories.append(len(values))
    return mappings, n_categories


def transform_geo(df: pd.DataFrame, mappings: dict[str, dict[int, int]]) -> np.ndarray:
    arrays = []
    for col in GEO_COLS:
        encoded = df[col].astype(int).map(mappings[col]).fillna(0).astype(np.int64).to_numpy()
        arrays.append(encoded)
    return np.stack(arrays, axis=1)


def encode_labels(train_y: pd.Series, val_y: pd.Series) -> tuple[np.ndarray, np.ndarray, dict[int, int], dict[int, int]]:
    labels = sorted(train_y.astype(int).unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return (
        train_y.astype(int).map(label_to_idx).to_numpy(dtype=np.int64),
        val_y.astype(int).map(label_to_idx).to_numpy(dtype=np.int64),
        label_to_idx,
        idx_to_label,
    )


def prepare_split(df: pd.DataFrame, args: argparse.Namespace) -> SplitData:
    train_df, val_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df[TARGET_COL],
    )

    excluded = {TARGET_COL, ID_COL, *GEO_COLS}
    dense_cols = [col for col in df.columns if col not in excluded]
    if not dense_cols:
        raise ValueError("Nessuna feature densa disponibile dopo l'esclusione di id/target/geo.")

    scaler = StandardScaler()
    x_dense_train = scaler.fit_transform(train_df[dense_cols]).astype(np.float32)
    x_dense_val = scaler.transform(val_df[dense_cols]).astype(np.float32)

    mappings, n_geo_categories = fit_geo_mappings(train_df)
    x_geo_train = transform_geo(train_df, mappings)
    x_geo_val = transform_geo(val_df, mappings)
    y_train, y_val, label_to_idx, idx_to_label = encode_labels(train_df[TARGET_COL], val_df[TARGET_COL])

    return SplitData(
        x_dense_train=x_dense_train,
        x_dense_val=x_dense_val,
        x_geo_train=x_geo_train,
        x_geo_val=x_geo_val,
        y_train=y_train,
        y_val=y_val,
        dense_cols=dense_cols,
        label_to_idx=label_to_idx,
        idx_to_label=idx_to_label,
        n_geo_categories=n_geo_categories,
    )


def make_loader(
    x_dense: np.ndarray,
    x_geo: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x_dense),
        torch.from_numpy(x_geo.astype(np.int64)),
        torch.from_numpy(y.astype(np.int64)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def class_weights(y_train: np.ndarray, mode: str, device: torch.device) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.bincount(y_train)
    weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def predict_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x_dense, x_geo, y in loader:
            x_dense = x_dense.to(device)
            x_geo = x_geo.to(device)
            logits = model(x_dense, x_geo)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.append(pred)
            y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def train_one_model(
    name: str,
    split: SplitData,
    args: argparse.Namespace,
    embedding_dims: list[int],
    hidden_dims: list[int],
    device: torch.device,
    use_dense: bool,
    use_geo: bool,
) -> tuple[dict, list[dict], np.ndarray]:
    model = TabularGeoNet(
        n_dense_features=split.x_dense_train.shape[1],
        n_geo_categories=split.n_geo_categories,
        embedding_dims=embedding_dims,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        n_classes=len(split.label_to_idx),
        use_dense=use_dense,
        use_geo=use_geo,
    ).to(device)

    train_loader = make_loader(
        split.x_dense_train,
        split.x_geo_train,
        split.y_train,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        split.x_dense_val,
        split.x_geo_val,
        split.y_val,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights(split.y_train, args.class_weight, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_f1 = -np.inf
    best_state = None
    stale_epochs = 0
    history = []
    start_total = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        losses = []

        for x_dense, x_geo, y in train_loader:
            x_dense = x_dense.to(device)
            x_geo = x_geo.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_dense, x_geo)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        y_train_true, y_train_pred = predict_model(model, train_loader, device)
        y_val_true, y_val_pred = predict_model(model, val_loader, device)
        train_f1 = f1_score(y_train_true, y_train_pred, average="micro")
        val_f1 = f1_score(y_val_true, y_val_pred, average="micro")
        val_macro = f1_score(y_val_true, y_val_pred, average="macro")

        row = {
            "model": name,
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "train_f1_micro": float(train_f1),
            "val_f1_micro": float(val_f1),
            "val_f1_macro": float(val_macro),
            "seconds": round(time.time() - epoch_start, 3),
        }
        history.append(row)
        print(
            f"{name} | epoch {epoch:02d} | loss={row['loss']:.4f} | "
            f"train_f1={train_f1:.4f} | val_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1 + 1e-5:
            best_f1 = val_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= args.patience:
            print(f"{name} | early stopping dopo {args.patience} epoche senza miglioramento.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_val_true, y_val_pred = predict_model(model, val_loader, device)
    result = {
        "model": name,
        "f1_micro": float(f1_score(y_val_true, y_val_pred, average="micro")),
        "f1_macro": float(f1_score(y_val_true, y_val_pred, average="macro")),
        "accuracy": float(accuracy_score(y_val_true, y_val_pred)),
        "best_epoch": int(max(history, key=lambda row: row["val_f1_micro"])["epoch"]),
        "seconds_total": round(time.time() - start_total, 3),
    }
    return result, history, y_val_pred


def evaluate_majority_baseline(split: SplitData) -> dict:
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(split.x_dense_train, split.y_train)
    pred = dummy.predict(split.x_dense_val)
    return {
        "model": "majority_baseline",
        "f1_micro": float(f1_score(split.y_val, pred, average="micro")),
        "f1_macro": float(f1_score(split.y_val, pred, average="macro")),
        "accuracy": float(accuracy_score(split.y_val, pred)),
        "best_epoch": 0,
        "seconds_total": 0.0,
    }


def decoded_report(y_true: np.ndarray, y_pred: np.ndarray, idx_to_label: dict[int, int]) -> str:
    y_true_decoded = np.array([idx_to_label[int(value)] for value in y_true], dtype=int)
    y_pred_decoded = np.array([idx_to_label[int(value)] for value in y_pred], dtype=int)
    return classification_report(y_true_decoded, y_pred_decoded, digits=4)


def main() -> None:
    args = parse_args()
    set_reproducibility(args.random_state)
    embedding_dims = parse_int_list(args.embedding_dims, expected_len=len(GEO_COLS))
    hidden_dims = parse_int_list(args.hidden_dims)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = PROJECT_ROOT / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PYTORCH GEO EMBEDDING FULL EXPERIMENT")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Torch: {torch.__version__}")

    df = sample_dataset(load_dataset(), args)
    print(f"Dataset usato: {df.shape[0]} righe x {df.shape[1]} colonne")
    print("Distribuzione classi:")
    print(df[TARGET_COL].value_counts().sort_index().to_string())

    split = prepare_split(df, args)
    print(f"\nFeature dense senza geo: {len(split.dense_cols)}")
    print("Categorie geo viste nel train:")
    for col, n_categories in zip(GEO_COLS, split.n_geo_categories):
        print(f"- {col}: {n_categories}")

    model_specs = {
        "no_geo": ("mlp_no_geo", True, False),
        "geo_only": ("geo_embedding_only", False, True),
        "with_geo": ("mlp_with_geo_embeddings", True, True),
    }

    results = [evaluate_majority_baseline(split)]
    histories = []
    predictions = {}

    for model_key in args.models:
        name, use_dense, use_geo = model_specs[model_key]
        print("\n" + "=" * 80)
        print(f"Training {name}")
        print("=" * 80)
        result, history, y_pred = train_one_model(
            name=name,
            split=split,
            args=args,
            embedding_dims=embedding_dims,
            hidden_dims=hidden_dims,
            device=device,
            use_dense=use_dense,
            use_geo=use_geo,
        )
        results.append(result)
        histories.extend(history)
        predictions[name] = y_pred

    results_df = pd.DataFrame(results).sort_values("f1_micro", ascending=False).reset_index(drop=True)
    history_df = pd.DataFrame(histories)

    results_path = output_dir / "geo_embedding_torch_results.csv"
    history_path = output_dir / "geo_embedding_torch_history.csv"
    summary_path = output_dir / "geo_embedding_torch_summary.json"

    results_df.to_csv(results_path, index=False)
    history_df.to_csv(history_path, index=False)

    best_model = str(results_df.iloc[0]["model"])
    best_report = ""
    if best_model in predictions:
        best_report = decoded_report(split.y_val, predictions[best_model], split.idx_to_label)

    summary = {
        "sample_mode": args.sample_mode,
        "n_rows": int(df.shape[0]),
        "test_size": float(args.test_size),
        "device": str(device),
        "torch_version": torch.__version__,
        "embedding_dims": embedding_dims,
        "hidden_dims": hidden_dims,
        "dropout": float(args.dropout),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "class_weight": args.class_weight,
        "dense_feature_count": int(len(split.dense_cols)),
        "n_geo_categories": dict(zip(GEO_COLS, split.n_geo_categories)),
        "results": results_df.to_dict(orient="records"),
        "best_model": best_model,
        "best_classification_report": best_report,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("RISULTATI FINALI")
    print("=" * 80)
    print(results_df.to_string(index=False))

    if best_report:
        print(f"\nClassification report miglior modello ({best_model}):")
        print(best_report)

    print("\nFile salvati:")
    print(f"- {results_path}")
    print(f"- {history_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
