"""
IDW (Inverse Distance Weighting) untuk mengisi piksel awan, PER KECAMATAN.

Alur singkat:
  1. test_idw()                   -> cari parameter terbaik (n, r, p)
  2. inverse_distance_weighting() -> isi piksel kosong dengan parameter itu

Arti parameter:
  r = radius tetangga (piksel), p = pangkat jarak, n = minimal tetangga valid
Tiap kecamatan diproses sendiri, jadi tidak ada nilai yang dipinjam dari
kecamatan lain (aman untuk split train/validation/test per kecamatan).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import convolve, binary_dilation
from rasterio.features import geometry_mask

NODATA = -9999          # nilai piksel kosong (tertutup awan)
MIN_VALID_TEST = 50     # kecamatan dengan piksel valid < 50 dilewati saat tuning

RADII = [1, 2, 3, 4, 5, 6]
POWERS = [1, 2, 3]
MIN_NS = [4, 8, 12]


# ------------------------------------------------------------
# A. Inti IDW
# ------------------------------------------------------------
def _idw(band, valid, r, p):
    """
    Untuk SETIAP piksel: rata-rata berbobot piksel valid di sekitarnya
    (jarak 1..r, bobot = 1 / jarak^p).
    Mengembalikan (prediction, count): nilai IDW dan jumlah tetangga valid.
    convolve dipakai supaya cepat; hasilnya sama dengan
    sum(bobot * nilai) / sum(bobot) untuk tiap piksel.
    """
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    distances = np.hypot(x, y)
    inside = (distances > 0) & (distances <= r)      # tetangga, bukan diri sendiri

    weights = np.zeros_like(distances, dtype=float)
    weights[inside] = 1 / distances[inside] ** p

    valid_float = valid.astype(float)
    numerator = convolve(np.where(valid, band, 0.0), weights, mode="constant", cval=0.0)
    denominator = convolve(valid_float, weights, mode="constant", cval=0.0)
    count = convolve(valid_float, inside.astype(float), mode="constant", cval=0.0)

    prediction = np.full(band.shape, np.nan)
    np.divide(numerator, denominator, out=prediction, where=denominator > 0)
    return prediction, count


# ------------------------------------------------------------
# B. Membagi citra menjadi kecamatan
# ------------------------------------------------------------
def _zones(items, boundary, only_names=None):
    """
    Daftar (nama, mask, window) untuk tiap kecamatan.
      mask   = piksel milik kecamatan itu
      window = kotak pembatas kecamatan (supaya hitungan lebih cepat)
    only_names: jika diisi, hanya kecamatan dengan nama itu.
    """
    boundary_crs = boundary.to_crs(items["crs"])
    shape = items["image"].shape[1:]
    zones = []

    for row in boundary_crs.itertuples():
        name = row.WADMKC
        if only_names is not None and name not in only_names:
            continue

        mask = geometry_mask(
            [row.geometry], transform=items["transform"], out_shape=shape, invert=True
        ) & items["mask_image"]
        if not mask.any():
            continue

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        window = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        zones.append((name, mask, window))

    return zones


# ------------------------------------------------------------
# C. Tuning parameter
# ------------------------------------------------------------
def _make_holes(valid, test_fraction, rng):
    """Buat lubang buatan 5x5 di piksel valid (untuk menguji akurasi IDW)."""
    seeds = valid & (rng.random(valid.shape) < test_fraction / 25)
    holes = binary_dilation(seeds, structure=np.ones((3, 3), dtype=bool), iterations=2)
    return holes & valid


def _score_all_params(band, valid, holes, scale):
    """Coba semua kombinasi (r, p, n); kembalikan daftar nRMSE dan coverage."""
    known = valid & ~holes              # piksel yang boleh dipakai sebagai tetangga
    actual = band[holes]
    rows = []

    for r in RADII:
        for p in POWERS:
            prediction, neighbor_count = _idw(band, known, r, p)
            pred = prediction[holes]
            count = neighbor_count[holes]
            squared_error = (actual - pred) ** 2

            for n in MIN_NS:
                ok = (count >= n) & np.isfinite(pred)
                if not ok.any():
                    continue
                rows.append({
                    "radius": r,
                    "power": p,
                    "minimum_n": n,
                    "nRMSE": np.sqrt(np.mean(squared_error[ok])) / scale,
                    "coverage_percent": 100 * ok.sum() / holes.sum(),
                })
    return rows


def _pick_best(results):
    """Ambil kombinasi dengan nRMSE terkecil di antara yang coverage-nya >= 90% dari terbaik."""
    if len(results) == 0:
        raise ValueError("IDW test produced no results. Check the valid pixels.")

    table = pd.DataFrame(results)
    table = table.groupby(["radius", "power", "minimum_n"], as_index=False).mean()

    threshold = 0.9 * table["coverage_percent"].max()
    eligible = table[table["coverage_percent"] >= threshold]
    best = eligible.loc[eligible["nRMSE"].idxmin()]

    return int(best["minimum_n"]), int(best["radius"]), int(best["power"]), table


def test_idw(images, boundary, train_names=None, test_fraction=0.1, seed=42):
    """
    Cari parameter IDW (n, r, p).
    train_names: nama kecamatan TRAIN. Jika diisi, tuning hanya memakai
    kecamatan itu (validation/test tidak ikut menentukan parameter).
    """
    rng = np.random.default_rng(seed)
    results = []

    for _, group_image in images.items():
        for items in group_image:
            for name, zone_mask, window in _zones(items, boundary, only_names=train_names):
                for band_full in items["image"]:
                    band = band_full[window].astype(float)
                    zone = zone_mask[window]

                    valid = (band != NODATA) & zone
                    if valid.sum() < MIN_VALID_TEST:
                        continue

                    scale = band[valid].std()
                    if scale == 0:
                        continue

                    holes = _make_holes(valid, test_fraction, rng)
                    if not holes.any():
                        continue

                    results += _score_all_params(band, valid, holes, scale)

    return _pick_best(results)


# ------------------------------------------------------------
# D. Mengisi piksel kosong
# ------------------------------------------------------------
def _fill_zone_band(band_view, zone, n, r, p):
    """Isi piksel kosong satu band di satu kecamatan. Mengembalikan (jumlah kosong, jumlah terisi)."""
    valid = (band_view != NODATA) & zone
    missing = zone & ~valid

    prediction, neighbor_count = _idw(band_view.astype(float), valid, r, p)

    fill = missing & (neighbor_count >= n) & np.isfinite(prediction)
    band_view[fill] = prediction[fill]      # menulis langsung ke citra asli
    return int(missing.sum()), int(fill.sum())


def inverse_distance_weighting(images, boundary, n, r, p, report_path=None):
    """
    Isi piksel NoData per kecamatan.
    report_path: jika diisi, simpan CSV laporan piksel terisi / masih kosong.
    Piksel yang masih -9999 harus dikeluarkan saat agregasi per kecamatan.
    """
    report = []

    for period, group_image in images.items():
        for items in group_image:
            # float agar hasil IDW tidak terpotong jadi bilangan bulat
            if not np.issubdtype(items["image"].dtype, np.floating):
                items["image"] = items["image"].astype("float32")

            for name, zone_mask, window in _zones(items, boundary):
                zone = zone_mask[window]

                for index in range(items["image"].shape[0]):
                    band_view = items["image"][index][window]
                    missing, filled = _fill_zone_band(band_view, zone, n, r, p)

                    report.append({
                        "period": period,
                        "subdistrict": name,
                        "band_index": index,
                        "zone_pixels": int(zone.sum()),
                        "missing_before": missing,
                        "filled": filled,
                        "still_missing": missing - filled,
                    })

    if report_path is not None:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(report).to_csv(report_path, index=False)

    return images