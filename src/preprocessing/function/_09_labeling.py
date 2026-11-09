import numpy as np
import pandas as pd


def normalize_lst_per_period(dataframe):
    dataframe = dataframe.copy()

    ref = {}
    for period in dataframe["period"].unique():
        rows = dataframe[dataframe["period"] == period]
        semua_piksel = np.concatenate(rows["lst"].values)
        ref[period] = np.nanmean(semua_piksel)

    hasil = []
    for i in range(len(dataframe)):
        period = dataframe["period"].iloc[i]
        arr = dataframe["lst"].iloc[i]
        hasil.append(arr - ref[period])

    dataframe["lst"] = hasil
    return dataframe


def fit_natural_breaks(dataframe, n_bins=2048, verbose=False):
    lst = np.concatenate(dataframe["lst"].values)

    if len(lst) == 0:
        raise ValueError("Tidak ada data valid.")

    hist, edge = np.histogram(lst, bins=n_bins)
    center = (edge[:-1] + edge[1:]) / 2

    mask = hist > 0
    hist = hist[mask].astype(float)
    center = center[mask]

    cum_w = np.cumsum(hist)
    cum_x = np.cumsum(hist * center)
    cum_x2 = np.cumsum(hist * center**2)

    total_w = cum_w[-1]
    total_x = cum_x[-1]
    total_x2 = cum_x2[-1]

    sdcm_min = np.inf
    b_b1 = None
    b_b2 = None

    m = len(hist)

    for i in range(m - 2):
        w1 = cum_w[i]
        x1 = cum_x[i]
        x21 = cum_x2[i]
        sdcm1 = x21 - x1**2 / w1

        j = np.arange(i + 1, m - 1)

        w2 = cum_w[j] - w1
        x2 = cum_x[j] - x1
        x22 = cum_x2[j] - x21

        w3 = total_w - cum_w[j]
        x3 = total_x - cum_x[j]
        x23 = total_x2 - cum_x2[j]

        total_sdcm = sdcm1 + (x22 - x2**2 / w2) + (x23 - x3**2 / w3)

        k = int(np.argmin(total_sdcm)) 
        if total_sdcm[k] < sdcm_min:
            sdcm_min = total_sdcm[k]
            b_b1 = center[i]
            b_b2 = center[j[k]]

    mean_all = np.average(center, weights=hist)
    sdam = np.sum(hist * (center - mean_all) ** 2)
    sdcm = sdcm_min

    if sdam > 0:
        gvf = (sdam - sdcm) / sdam
    else:
        gvf = np.nan

    if verbose:
        print("SDAM:", sdam)
        print("SDCM:", sdcm)
        print("GVF:", gvf)
        print("Batas kelas rendah:", b_b1)
        print("Batas kelas tinggi:", b_b2)

    return {"b1": float(b_b1), "b2": float(b_b2), "gvf": float(gvf)}


def apply_natural_breaks(dataframe, breaks):
    b1, b2 = breaks["b1"], breaks["b2"]
    label_total = []

    for lst in dataframe["lst"]:
        count_rendah = np.sum(lst <= b1)
        count_tinggi = np.sum(lst > b2)
        count_normal = len(lst) - count_rendah - count_tinggi

        if (count_rendah >= count_normal) and (count_rendah >= count_tinggi):
            label_total.append(0)
        elif (count_normal >= count_rendah) and (count_normal >= count_tinggi):
            label_total.append(1)
        else:
            label_total.append(2)

    return pd.Series(label_total, index=dataframe.index, name="lst_label")