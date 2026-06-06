from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

GEO_COLS = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]


class GeoLevelClassifier(nn.Module):
    def __init__(
        self,
        n_geo_categories: list[int],
        embedding_dims: list[int],
        hidden_dim: int,
        dropout: float,
        n_classes: int,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=n + 1, embedding_dim=d, padding_idx=0)
                for n, d in zip(n_geo_categories, embedding_dims)
            ]
        )
        self.encoder = nn.Sequential(
            nn.Linear(sum(embedding_dims), hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x_geo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = [embedding(x_geo[:, i]) for i, embedding in enumerate(self.embeddings)]
        x = torch.cat(embeddings, dim=1)
        code = self.encoder(x)
        logits = self.classifier(code)
        return logits, code


def fit_geo_mappings(df: pd.DataFrame) -> tuple[Dict[str, Dict[int, int]], list[int]]:
    mappings: dict[str, dict[int, int]] = {}
    n_geo_categories: list[int] = []
    for col in GEO_COLS:
        values = sorted(df[col].astype(int).dropna().unique().tolist())
        mappings[col] = {value: idx + 1 for idx, value in enumerate(values)}
        n_geo_categories.append(len(values))
    return mappings, n_geo_categories


def transform_geo(df: pd.DataFrame, mappings: dict[str, dict[int, int]]) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for col in GEO_COLS:
        encoded = (
            df[col]
            .astype(float)
            .map(mappings[col])
            .fillna(0)
            .astype(np.int64)
            .to_numpy()
        )
        arrays.append(encoded)
    return np.stack(arrays, axis=1)


def _make_loader(x_geo: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x_geo.astype(np.int64)), torch.from_numpy(y.astype(np.int64)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_geo_hidden_features(
    train_geo_df: pd.DataFrame,
    train_y: np.ndarray,
    test_geo_df: pd.DataFrame,
    hidden_dim: int,
    embedding_dims: list[int] = (4, 8, 16),
    dropout: float = 0.1,
    epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    device: str | torch.device = "cpu",
    compute_test: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[int, int]]]:
    device = torch.device(device)

    mappings, n_geo_categories = fit_geo_mappings(train_geo_df)
    x_geo_train = transform_geo(train_geo_df, mappings)
    if compute_test:
        x_geo_test = transform_geo(test_geo_df, mappings)
    else:
        x_geo_test = None

    unique_labels = sorted(np.unique(train_y))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y_train = np.array([label_to_idx[label] for label in train_y], dtype=np.int64)
    n_classes = len(unique_labels)

    model = GeoLevelClassifier(
        n_geo_categories=n_geo_categories,
        embedding_dims=list(embedding_dims),
        hidden_dim=hidden_dim,
        dropout=dropout,
        n_classes=n_classes,
    ).to(device)

    loader = _make_loader(x_geo_train, y_train, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_loss = float("inf")
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for x_geo, y in loader:
            x_geo = x_geo.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x_geo)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        avg_loss = float(np.mean(losses)) if losses else 0.0
        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        x_geo_train_t = torch.from_numpy(x_geo_train.astype(np.int64)).to(device)
        _, hidden_train = model(x_geo_train_t)
        hidden_test = None
        if compute_test and x_geo_test is not None and x_geo_test.size > 0:
            x_geo_test_t = torch.from_numpy(x_geo_test.astype(np.int64)).to(device)
            _, hidden_test = model(x_geo_test_t)

    hidden_train = hidden_train.cpu().numpy()
    if hidden_test is not None:
        hidden_test = hidden_test.cpu().numpy()

    def append_hidden(df: pd.DataFrame, hidden_vectors: np.ndarray) -> pd.DataFrame:
        df_out = df.reset_index(drop=True).copy()
        for idx in range(hidden_vectors.shape[1]):
            df_out[f"geo_hidden_{idx}"] = hidden_vectors[:, idx]
        return df_out

    train_with_hidden = append_hidden(train_geo_df, hidden_train)
    if hidden_test is not None:
        test_with_hidden = append_hidden(test_geo_df, hidden_test)
    else:
        test_with_hidden = test_geo_df.reset_index(drop=True).copy()

    return train_with_hidden, test_with_hidden, mappings
