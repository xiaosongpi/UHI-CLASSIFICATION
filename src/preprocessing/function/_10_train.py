import json
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from joblib import Parallel, delayed

from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (classification_report, confusion_matrix)

from function._09_labeling import fit_natural_breaks, apply_natural_breaks
from function._11_model import FTTransformer

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

LABEL_MODE = "global"

PARAM = {
    "learning_rate": [0.00005, 0.0001, 0.0003],
    "n_transformer": [1, 2, 3],
    "d_model": [64, 96, 128]
}

FEATURES = [
    "ndvi_mean", "ndvi_std", "ndvi_min", "ndvi_max", "ndvi_q1", "ndvi_median", "ndvi_q3",
    "ndbi_mean", "ndbi_std", "ndbi_min", "ndbi_max", "ndbi_q1", "ndbi_median", "ndbi_q3",
    "mndwi_mean", "mndwi_std", "mndwi_min", "mndwi_max", "mndwi_q1", "mndwi_median", "mndwi_q3"
]

_EKSTRA = ("mode", "ref", "cadangan", "per_period", "feat_ref") 


def _suffix():
    return "" if LABEL_MODE == "global" else f"_{LABEL_MODE}"


def _pkey(series):
    return series.astype(str)

def _normalisasi(df, ref):
    d = df.copy()
    d["lst"] = pd.Series(
        [arr - ref[str(p)] for arr, p in zip(df["lst"], df["period"])],
        index=df.index,
        dtype=object,
    )
    return d


def _feat_ref(df):
    return (
        df[FEATURES]
        .groupby(_pkey(df["period"]))
        .mean()
        .to_dict("index")
    )


def fit_labels(df, verbose=False):
    if LABEL_MODE == "global":
        return fit_natural_breaks(df, verbose=verbose)

    elif LABEL_MODE == "normalisasi":
        ref = {
            str(p): float(np.mean(np.concatenate(g["lst"].values)))
            for p, g in df.groupby("period")
        }
        breaks = dict(fit_natural_breaks(_normalisasi(df, ref), verbose=verbose))
        breaks["mode"] = "normalisasi"
        breaks["ref"] = ref
        breaks["feat_ref"] = _feat_ref(df)
        return breaks

    elif LABEL_MODE == "per_periode":
        breaks = dict(fit_natural_breaks(df, verbose=verbose))
        breaks["mode"] = "per_periode"
        breaks["per_period"] = {
            str(p): fit_natural_breaks(g) for p, g in df.groupby("period")
        }
        breaks["feat_ref"] = _feat_ref(df)
        return breaks

    else:
        raise ValueError(f"LABEL_MODE tidak dikenal: {LABEL_MODE}")


def apply_labels(df, breaks):
    bersih = {k: v for k, v in breaks.items() if k not in _EKSTRA}

    if LABEL_MODE == "global":
        return apply_natural_breaks(df, breaks)

    elif LABEL_MODE == "normalisasi":
        d = _normalisasi(df, breaks["ref"])
        return apply_natural_breaks(d, bersih)

    elif LABEL_MODE == "per_periode":
        out = pd.Series(0, index=df.index, dtype="int64")
        for p, g in df.groupby("period"):
            b = breaks["per_period"][str(p)]
            out.loc[g.index] = apply_natural_breaks(g, b)
        return out

    else:
        raise ValueError(f"LABEL_MODE tidak dikenal: {LABEL_MODE}")

def run_fold(params, fold):
    torch.set_num_threads(1) 
    x_tr, y_tr, x_va, y_va = fold
    return train_model(x_tr, y_tr, x_va, y_va, params)

def standart_data(x_train, x_validation):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0, ddof=0)

    std = std.mask(std == 0, 1.0)

    x_train_s = (x_train - mean) / std
    x_validation_s = (x_validation - mean) / std

    return x_train_s, x_validation_s, mean, std

def _prov_labels(df):
    return apply_natural_breaks(df, fit_natural_breaks(df))

def split_data(dataset):
    x = dataset[FEATURES]

    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Fitur mengandung NaN atau infinity.")

    if (x == -9999).any().any():
        raise ValueError("Fitur masih mengandung sentinel -9999.")

    groups = dataset["subdistrict"].astype(str)
    y_prov = _prov_labels(dataset)

    s1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, temp_idx = next(s1.split(dataset, y_prov, groups))

    train_df = dataset.iloc[train_idx]
    temp_df = dataset.iloc[temp_idx]

    s2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
    val_idx, test_idx = next(
        s2.split(
            temp_df,
            y_prov.loc[temp_df.index],
            temp_df["subdistrict"].astype(str)
        )
    )

    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]

    assert set(train_df["subdistrict"]).isdisjoint(
        set(val_df["subdistrict"])
    )
    assert set(train_df["subdistrict"]).isdisjoint(
        set(test_df["subdistrict"])
    )
    assert set(val_df["subdistrict"]).isdisjoint(
        set(test_df["subdistrict"])
    )

    return train_df, val_df, test_df


def make_xy(dataframe, breaks):
    """Fitur + label, memakai batas yang sudah ditentukan dari data latih."""
    x = dataframe[FEATURES].copy()

    if LABEL_MODE != "global":
        ref = pd.DataFrame(breaks["feat_ref"]).T[FEATURES]   
        x = x - ref.loc[_pkey(dataframe["period"])].to_numpy()

    y = apply_labels(dataframe, breaks)
    return x, y


def convert_tensor(x, y):
    x_tensor = torch.tensor(
        x.to_numpy(dtype = np.float32),
        dtype=torch.float32,
    )
    y_tensor = torch.tensor(
        y.to_numpy(dtype = np.int64),
        dtype=torch.long,
    )

    return x_tensor, y_tensor


def count_metrics(y_true, predictions):
    matrix = np.zeros((3, 3))
    np.add.at(matrix, (y_true, predictions), 1)

    tp = np.diag(matrix)
    predicted_total = matrix.sum(axis = 0)
    actual_total = matrix.sum(axis = 1)

    precision = np.divide(tp, predicted_total, out = np.zeros(3), where = predicted_total > 0)
    recall = np.divide(tp, actual_total, out = np.zeros(3), where = actual_total > 0)

    f1_total = precision + recall
    f1 = np.divide(2 * precision * recall, f1_total, out = np.zeros(3), where = f1_total > 0)

    return {
        "accuracy": tp.sum() / matrix.sum(),
        "precision": precision.mean(),
        "recall": recall.mean(),
        "f1_score": f1.mean(),
    }

def _flatten_breaks(b):
    flat = {k: v for k, v in b.items() if k not in _EKSTRA and k != "mode"}
    for p, v in b.get("ref", {}).items():
        flat[f"ref_{p}"] = v
    for p, v in b.get("per_period", {}).items():
        flat[f"b1_{p}"] = v["b1"]
        flat[f"b2_{p}"] = v["b2"]
    return flat


def _baseline_majority(y_train, y_true):
    mayoritas = int(y_train.value_counts().idxmax())
    pred = np.full(len(y_true), mayoritas, dtype=int)
    m = count_metrics(y_true, pred)
    return {"kelas": mayoritas, **{k: float(v) for k, v in m.items()}}


def train_model(
    x_train, y_train,
    x_validation, y_validation,
    params,
    return_model=False
):
    torch.manual_seed(42)
    np.random.seed(42)

    epochs = 200
    patience = 30
    min_delta = 0.0001

    x_train_s, x_validation_s, mean, std = standart_data(
        x_train, x_validation
    )

    x_train_t, y_train_t = convert_tensor(x_train_s, y_train)
    x_validation_t, y_validation_t = convert_tensor(
        x_validation_s, y_validation
    )

    batch_data = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=32,
        shuffle=True,
    )

    model = FTTransformer(
        n_feature=21,
        n_class=3,
        d_model=params["d_model"],
        n_transformer=params["n_transformer"],
        n_head=8,
        dropout=0.1,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if (
            "weight" in name
            and "norm" not in name.lower()
            and name not in ["weight", "bias"]
        ):
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 1e-5},
            {"params": no_decay_params, "weight_decay": 0.0}
        ],
        lr=params["learning_rate"]
    )

    best_val_loss = float("inf")
    best_model_state = None
    counter = 0
    best_epoch = 0

    start_time = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        train_loss_total = 0.0

        for x_b, y_b in batch_data:
            x_b = x_b.to(DEVICE)
            y_b = y_b.to(DEVICE)

            optimizer.zero_grad()

            logits = model(x_b)
            loss = criterion(logits, y_b)

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

        avg_train_loss = train_loss_total / len(batch_data)

        # Validasi
        model.eval()

        with torch.no_grad():
            val_logits = model(x_validation_t.to(DEVICE))
            val_loss = criterion(
                val_logits,
                y_validation_t.to(DEVICE)
            ).item()

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            counter = 0

        else:
            counter += 1

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Patience: {counter}/{patience}",
            flush=True,
        )

        if counter >= patience:
            print(
                f"\nEarly Stopping pada epoch {epoch + 1}. "
                f"Best Epoch: {best_epoch}, "
                f"Best Val Loss: {best_val_loss:.4f}",
                flush=True,
            )
            break

    if best_model_state is None:
        raise RuntimeError("Training gagal menghasilkan model terbaik.")

    model.load_state_dict(best_model_state)

    training_time = time.perf_counter() - start_time

    model.eval()

    with torch.no_grad():
        logits = model(x_validation_t.to(DEVICE))
        probabilities = torch.softmax(
            logits, dim=1
        ).cpu().numpy()

        predictions = probabilities.argmax(axis=1)

    y_true = y_validation_t.numpy()

    metrics = count_metrics(y_true, predictions)
    metrics["training_time"] = training_time
    metrics["best_epoch"] = best_epoch
    metrics["best_val_loss"] = best_val_loss

    if return_model:
        return (
            metrics,
            model,
            mean,
            std,
            predictions,
            probabilities
        )

    return metrics


def build_folds(train_df, n_splits=5):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    groups = train_df["subdistrict"].astype(str)
    y_prov = _prov_labels(train_df)

    folds = []
    fold_breaks = []

    if train_df["subdistrict"].nunique() < n_splits:
        raise ValueError("Jumlah kecamatan training kurang dari jumlah fold.")

    for k, (tr_idx, va_idx) in enumerate(
        cv.split(train_df, y_prov, groups), start=1
    ):
        tr_df = train_df.iloc[tr_idx]
        va_df = train_df.iloc[va_idx]

        breaks = fit_labels(tr_df)
        x_tr, y_tr = make_xy(tr_df, breaks)
        x_va, y_va = make_xy(va_df, breaks)

        print(
            f"Fold {k}: b1={breaks['b1']:.4f} b2={breaks['b2']:.4f} "
            f"gvf={breaks['gvf']:.4f} | "
            f"latih={y_tr.value_counts().sort_index().to_dict()} "
            f"valid={y_va.value_counts().sort_index().to_dict()}",
            flush=True,
        )

        if y_tr.nunique() < 3 or y_va.nunique() < 3:
            print(f"PERINGATAN fold {k}: ada kelas yang hilang "
                  f"(latih={y_tr.nunique()} kelas, valid={y_va.nunique()} kelas)",
                  flush=True)

        folds.append((x_tr, y_tr, x_va, y_va))
        fold_breaks.append({"fold": k, **breaks})

    return folds, fold_breaks


def search_best_params(train_df, n_jobs=4):
    n_splits = 5
    folds, fold_breaks = build_folds(train_df, n_splits)

    flat_df = pd.DataFrame([{"fold": b["fold"], **_flatten_breaks(b)} for b in fold_breaks])
    flat_df.to_csv(OUTPUT_DIR / f"breaks_per_fold{_suffix()}.csv", index=False)
    with open(OUTPUT_DIR / f"breaks_per_fold{_suffix()}.json", "w") as f:
        json.dump(fold_breaks, f, indent=2, default=float)

    print("\nStabilitas batas antar fold (mean, std):")
    print(flat_df.drop(columns="fold").agg(["mean", "std"]).T.to_string(), flush=True)

    combinations = [
        {"learning_rate": lr, "n_transformer": n, "d_model": d}
        for lr in PARAM["learning_rate"]
        for n in PARAM["n_transformer"]
        for d in PARAM["d_model"]
    ]

    all_metrics = Parallel(n_jobs=n_jobs)(
        delayed(run_fold)(p, fold)
        for p in combinations
        for fold in folds
    )

    metric_names = ["accuracy", "precision", "recall", "f1_score", "training_time"]
    best_score, best_params, results = -1, None, []

    for i, params in enumerate(combinations):
        fold_metrics = all_metrics[i * n_splits:(i + 1) * n_splits]
        mean_results = {
            f"mean_{k}": np.mean([m[k] for m in fold_metrics]) for k in metric_names
        }
        results.append({**params, **mean_results})

        if mean_results["mean_f1_score"] > best_score:
            best_score = mean_results["mean_f1_score"]
            best_params = params.copy()

    return best_params, results


def train_final_model(dataset, label_mode=None):

    global LABEL_MODE
    if label_mode is not None:
        LABEL_MODE = label_mode
    print("Mode pelabelan:", LABEL_MODE)

    dataset = dataset.dropna().reset_index(drop=True)

    # [DITAMBAH] pastikan tidak ada baris tanpa piksel LST
    assert dataset["lst"].map(len).min() > 0, "Ada baris dengan LST kosong."

    train_df, val_df, test_df = split_data(dataset)

    print("Jumlah data train:", len(train_df))
    print("Jumlah data validation:", len(val_df))
    print("Jumlah data test:", len(test_df))

    if LABEL_MODE != "global":
        p_latih = set(train_df["period"].astype(str))
        p_lain = set(val_df["period"].astype(str)) | set(test_df["period"].astype(str))
        if not p_lain.issubset(p_latih):
            raise ValueError(f"Periode tidak ada di data latih: {p_lain - p_latih}")

    best_params, cv_results = search_best_params(train_df)

    print("Best params:", best_params)

    final_breaks = fit_labels(train_df, verbose=True)

    x_train, y_train = make_xy(train_df, final_breaks)
    x_val, y_val = make_xy(val_df, final_breaks)
    x_test, y_test = make_xy(test_df, final_breaks)

    print("\nDistribusi kelas TRAIN:")
    print(y_train.value_counts().sort_index())

    print("\nDistribusi kelas VALIDATION:")
    print(y_val.value_counts().sort_index())

    print("\nDistribusi kelas TEST:")
    print(y_test.value_counts().sort_index())

    print("\nProporsi kelas per periode (TRAIN):")
    print(
        pd.crosstab(_pkey(train_df["period"]), y_train, normalize="index").round(3)
    )

    val_metrics, model, mean, std, _, _ = train_model(
        x_train, y_train,
        x_val, y_val,
        best_params,
        return_model=True,
    )

    x_test_s = (x_test - mean) / std

    x_test_t, y_test_t = convert_tensor(
        x_test_s, y_test
    )

    model.eval()

    with torch.no_grad():
        logits = model(x_test_t.to(DEVICE))

        probabilities = torch.softmax(
            logits, dim=1
        ).cpu().numpy()

        predictions = probabilities.argmax(axis=1)

    y_true = y_test_t.numpy()
    class_names = ["Rendah", "Normal", "Tinggi"]

    metrics = count_metrics(y_true, predictions)

    baseline = _baseline_majority(y_train, y_true)

    print("\nTest metrics:", metrics)
    print("Baseline kelas mayoritas:", baseline)

    print(
        classification_report(
            y_true,
            predictions,
            labels=[0, 1, 2],
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1, 2],
    )

    print("Confusion Matrix:")
    print(matrix)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "params": best_params,
            "mean": mean.to_dict(),
            "std": std.to_dict(),
            "breaks": final_breaks, 
            "label_mode": LABEL_MODE,
        },
        OUTPUT_DIR / f"ft_transformer_final{_suffix()}.pt",
    )

    report = classification_report(
        y_true,
        predictions,
        labels=[0, 1, 2],
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    evaluation = {
        "label_mode": LABEL_MODE,
        "confusion_matrix": matrix.tolist(),
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["precision"]),
        "macro_recall": float(metrics["recall"]),
        "macro_f1": float(metrics["f1_score"]),
        "baseline_majority": baseline,
        "n_test": int(len(y_true)),
        "breaks": final_breaks,
        "best_params": best_params,
        "best_epoch": int(val_metrics["best_epoch"]),
        "best_val_loss": float(val_metrics["best_val_loss"]),
        "training_time": float(val_metrics["training_time"]),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1_score": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in class_names
        },
    }

    with open(OUTPUT_DIR / f"evaluasi_model{_suffix()}.json", "w") as f:
        json.dump(evaluation, f, indent=2, default=float)

    x_all, y_all = make_xy(dataset, final_breaks)
    x_all_s = (x_all - mean) / std

    x_all_t = torch.tensor(
        x_all_s.to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )

    model.eval()

    with torch.no_grad():
        prob_all = torch.softmax(
            model(x_all_t.to(DEVICE)),
            dim=1,
        ).cpu().numpy()

    pred_all = prob_all.argmax(axis=1)

    output = dataset[["subdistrict", "period"]].copy()

    output["kelas_sebenarnya"] = y_all.map(
        dict(enumerate(class_names))
    )

    output["kelas_prediksi"] = pd.Series(
        pred_all, index=dataset.index
    ).map(dict(enumerate(class_names)))

    output["prob_rendah"] = prob_all[:, 0]
    output["prob_normal"] = prob_all[:, 1]
    output["prob_tinggi"] = prob_all[:, 2]

    output["split"] = "latih"
    output.loc[val_df.index, "split"] = "validasi"
    output.loc[test_df.index, "split"] = "uji"

    output.to_csv(
        OUTPUT_DIR / f"prediksi_kecamatan{_suffix()}.csv",
        index=False,
    )

    cv_df = pd.DataFrame(cv_results)

    cv_df.to_csv(
        OUTPUT_DIR / f"hasil_grid_search{_suffix()}.csv",
        index=False,
    )

    return model, best_params, metrics, cv_df