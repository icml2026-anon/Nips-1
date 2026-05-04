import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from pathlib import Path


class SparseSequenceDataset(Dataset):

    def __init__(self, sequences, labels, masks=None, transform_fn=None):
        if isinstance(sequences, np.ndarray):
            self.sequences = torch.tensor(sequences, dtype=torch.float32)
        else:
            self.sequences = sequences.float()
        if isinstance(labels, np.ndarray):
            self.labels = torch.tensor(labels, dtype=torch.float32)
        else:
            self.labels = labels.float()
        if masks is not None:
            if isinstance(masks, np.ndarray):
                self.masks = torch.tensor(masks, dtype=torch.float32)
            else:
                self.masks = masks.float()
        else:
            self.masks = None
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.sequences[idx]
        y = self.labels[idx]
        if self.masks is not None:
            mask = self.masks[idx]
        else:
            if x.dim() == 1:
                mask = (x > 0).float()
            else:
                mask = (x.abs().sum(dim=-1) > 0).float()
        if self.transform_fn is not None:
            x = self.transform_fn(x)
        return {"ts_data": x, "mask": mask, "label": y}


def log_transform(x):
    return torch.log1p(x.clamp(min=0))


def standardize(x):
    if x.dim() == 1:
        std = x.std()
        if std > 1e-6:
            x = (x - x.mean()) / std
    elif x.dim() == 2:
        for c in range(x.shape[0]):
            std = x[c].std()
            if std > 1e-6:
                x[c] = (x[c] - x[c].mean()) / std
    return x


def default_transform(x):
    x = log_transform(x)
    x = standardize(x)
    return x


def load_dataset(dataset_name, data_root="./data", task=None):
    data_root = Path(data_root)

    if dataset_name == "merchant":
        import pandas as pd
        csv_path = data_root / "data" / "merchant_data.csv"
        df = pd.read_csv(csv_path)
        if task is not None and task.startswith("Industry"):
            df = df[df["Industry"] == task].reset_index(drop=True)
        txn_cols = sorted([c for c in df.columns if c.startswith("txn_")])
        cnt_cols = sorted([c for c in df.columns if c.startswith("cnt_")])
        if cnt_cols:
            amount = df[txn_cols].fillna(0).values.astype(np.float32)
            count  = df[cnt_cols].fillna(0).values.astype(np.float32)
            X = np.stack([amount, count], axis=-1)
        else:
            X = df[txn_cols].fillna(0).values.astype(np.float32)
        y = df["is_anomalous"].values.astype(np.float32)
        return X, y

    elif dataset_name == "retail":
        base = data_root / "datasets"
        if task == "churn":
            d = base / "retail_processed" / "churn_task"
        else:
            d = base / "retail_processed"
        amount = np.load(d / "amount_series.npy").astype(np.float32)
        trans = np.load(d / "trans_series.npy").astype(np.float32)
        labels = np.load(d / "labels.npy").astype(np.float32)
        X = np.stack([amount, trans], axis=-1)
        return X, labels

    elif dataset_name == "cdnow":
        base = data_root / "datasets"
        if task == "churn":
            d = base / "cdnow_processed" / "churn_task"
        else:
            d = base / "cdnow_processed"
        amount = np.load(d / "amount_series.npy").astype(np.float32)
        trans = np.load(d / "trans_series.npy").astype(np.float32)
        labels = np.load(d / "labels.npy").astype(np.float32)
        X = np.stack([amount, trans], axis=-1)
        return X, labels

    elif dataset_name == "instacart":
        base = data_root / "datasets"
        if task == "churn":
            d = base / "instacart_processed" / "churn_task"
        else:
            d = base / "instacart_processed"
        orders = np.load(d / "order_count_series.npy").astype(np.float32)
        items = np.load(d / "item_count_series.npy").astype(np.float32)
        labels = np.load(d / "labels.npy").astype(np.float32)
        X = np.stack([orders, items], axis=-1)
        return X, labels

    elif dataset_name == "sales_weekly":
        base = data_root / "datasets"
        if task == "seasonality":
            d = base / "sales_weekly_processed" / "seasonality_task"
        else:
            d = base / "sales_weekly_processed"
        sales = np.load(d / "sales_series.npy").astype(np.float32)
        labels = np.load(d / "labels.npy").astype(np.float32)
        X = sales[..., np.newaxis] if sales.ndim == 2 else sales
        return X, labels

    elif dataset_name == "tafeng":
        if task == "repurchase":
            d = data_root / "data" / "tafeng_repurchase_task"
        else:
            d = data_root / "data"
            amount = np.load(d / "tafeng_amount_series.npy").astype(np.float32)
            trans = np.load(d / "tafeng_trans_series.npy").astype(np.float32)
            labels = np.load(d / "tafeng_labels.npy").astype(np.float32)
            X = np.stack([amount, trans], axis=-1)
            return X, labels
        amount = np.load(d / "amount_series.npy").astype(np.float32)
        trans = np.load(d / "trans_series.npy").astype(np.float32)
        labels = np.load(d / "labels.npy").astype(np.float32)
        X = np.stack([amount, trans], axis=-1)
        return X, labels

    elif dataset_name == "physionet2012":
        base = data_root / "datasets"
        if task == "longstay":
            d = base / "physionet2012_processed" / "longstay_task"
        else:
            d = base / "physionet2012_processed"
        X = np.load(d / "vitals_series.npy").astype(np.float32)
        labels = np.load(d / "labels.npy").astype(np.float32)
        return X, labels

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def preprocess_splits(X_train, X_val, X_test):
    def _masks(X):
        if X.ndim == 2:
            return (X > 0).astype(np.float32)
        return (np.abs(X).sum(axis=-1) > 0).astype(np.float32)

    m_train = _masks(X_train)
    m_val = _masks(X_val)
    m_test = _masks(X_test)

    def _process(X, scale):
        X_log = np.log1p(np.clip(X, 0, None)).astype(np.float32)
        if X_log.ndim == 2:
            X_log = X_log[:, :, np.newaxis]

        mu = X_log.mean(axis=1, keepdims=True)
        std = np.clip(X_log.std(axis=1, keepdims=True), 1e-6, None)
        X_z = ((X_log - mu) / std).astype(np.float32)

        X_s = (X_log / scale).astype(np.float32)

        return np.concatenate([X_z, X_s], axis=-1)

    X_log_tr = np.log1p(np.clip(X_train, 0, None)).astype(np.float32)
    if X_log_tr.ndim == 2:
        X_log_tr = X_log_tr[:, :, np.newaxis]
    scale = np.clip(X_log_tr.max(axis=(0, 1)), 1e-6, None)

    X_tr = _process(X_train, scale)
    X_vl = _process(X_val, scale)
    X_te = _process(X_test, scale)

    return X_tr, X_vl, X_te, m_train, m_val, m_test


def prepare_splits(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y
    )
    relative_val = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val,
        random_state=seed, stratify=y_train_val
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
