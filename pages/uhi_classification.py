import calendar
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.services.historry_services import create_history
from src.services.gee_services import save_landsat_images
from src.preprocessing.function._01_boundary import get_boundary_file
from src.preprocessing.function._02_clipping import clipping_jakarta_boundary
from src.preprocessing.function._03_cloud_mask import cloud_masking
from src.preprocessing.function._04_rescalling import rescalling_ST_SR
from src.preprocessing.function._06_indeks_spektral import index_spectral
from src.preprocessing.function._07_composit_monthly import composite_image
from src.preprocessing.function._05_inverse_distance_weighting import inverse_distance_weighting
from src.preprocessing.function._08_statistic_count import count_statistic
from src.preprocessing.function._11_model import FTTransformer

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "src" / "preprocessing" / "models" / "ft_transformer_final.pt"
BOUNDARY_PATH = BASE_DIR / "data" / "raw_data" / "boundary.gpkg"

ALL_SUBDISTRICT = "Seluruh kecamatan"
MIN_SUBDISTRICT_PER_PERIOD = 10

missing = [p for p in (MODEL_PATH, BOUNDARY_PATH) if not p.exists()]
if missing:
    st.error(
        "File berikut tidak ditemukan di server:\n"
        + "\n".join(f"- {p.relative_to(BASE_DIR)}" for p in missing)
    )
    st.stop()

if BOUNDARY_PATH.stat().st_size < 1024:
    st.error(
        "boundary.gpkg terlalu kecil. Kemungkinan itu file pointer Git LFS, "
        "bukan data aslinya."
    )
    st.stop()

CLASS_NAMES = ["Rendah", "Normal", "Tinggi"]

FEATURES = [
    "ndvi_mean", "ndvi_std", "ndvi_min", "ndvi_max",
    "ndvi_q1", "ndvi_median", "ndvi_q3",
    "ndbi_mean", "ndbi_std", "ndbi_min", "ndbi_max",
    "ndbi_q1", "ndbi_median", "ndbi_q3",
    "mndwi_mean", "mndwi_std", "mndwi_min", "mndwi_max",
    "mndwi_q1", "mndwi_median", "mndwi_q3",
]

MONTH = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember",
]
YEAR = list(range(date.today().year, 2012, -1))

st.title("Klasifikasi UHI")
st.caption("Mulai mengklasifikasi tingkat UHI dimulai dari 42 kecamatan di DKI Jakarta, dengan memilih kecamatan yang ingin diklasifikasi serta memilih tanggal dan bulan yang ingin di klasifikasikan")
st.caption("Kelas UHI bersifat relatif: dibandingkan dengan kecamatan lain di Jakarta pada bulan yang sama.")


@st.cache_data
def load_subdistrict_names():
    boundary = get_boundary_file(str(BOUNDARY_PATH))
    names = sorted(boundary["WADMKC"].dropna().unique().tolist())
    return [ALL_SUBDISTRICT] + names


@st.cache_resource
def load_model():
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    params = ckpt["params"]
    model = FTTransformer(
        d_model=params["d_model"], n_transformer=params["n_transformer"]
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mean = pd.Series(ckpt["mean"]).reindex(FEATURES)
    std = pd.Series(ckpt["std"]).reindex(FEATURES)
    return model, mean, std


def predict_uhi(stat_df):
    model, mean, std = load_model()

    missing_cols = [col for col in FEATURES if col not in stat_df.columns]
    if missing_cols:
        raise ValueError(f"Kolom statistik tidak ada: {missing_cols}")

    x_df = stat_df[FEATURES].apply(pd.to_numeric, errors="coerce")
    x_df = x_df.replace([np.inf, -np.inf], np.nan).dropna()

    if x_df.empty:
        raise ValueError("Tidak ada data statistik yang valid.")

    periods = stat_df.loc[x_df.index, "period"].astype(str)
    if periods.groupby(periods).size().min() < MIN_SUBDISTRICT_PER_PERIOD:
        raise ValueError(
            "Jumlah kecamatan dengan data valid terlalu sedikit untuk dijadikan "
            "referensi periode (kemungkinan tertutup awan). Coba bulan lain."
        )
    x_df = x_df - x_df.groupby(periods).transform("mean")

    x = ((x_df - mean) / std).to_numpy(dtype=np.float32)

    with torch.inference_mode():
        prob = torch.softmax(model(torch.tensor(x)), dim=1).numpy()

    result = stat_df.loc[x_df.index].drop(columns=["lst"], errors="ignore")
    result["kelas_uhi"] = [CLASS_NAMES[i] for i in prob.argmax(axis=1)]
    for i, name in enumerate(CLASS_NAMES):
        result[f"prob_{name.lower()}"] = prob[:, i]
    return result


def reset_form():
    today = date.today()
    st.session_state["subdistrict"] = ALL_SUBDISTRICT
    st.session_state["year"] = today.year
    st.session_state["month"] = today.month


user = st.session_state.get("user")
if not user:
    if not st.session_state.pop("login_redirected", False):
        st.session_state["page"] = "login"
        st.session_state["open_dialog"] = True
        st.session_state["login_redirected"] = True
        st.rerun()
    st.stop()

left, right = st.columns(2, gap="large")
result_box = right.empty()

with left:
    with st.form("form", enter_to_submit=False):
        subdistrict = st.selectbox("Kecamatan", load_subdistrict_names(), key="subdistrict")
        col_1, col_2 = st.columns(2)
        month = col_1.selectbox("Bulan", range(1, 13), format_func=lambda m: MONTH[m - 1], key="month")
        year = col_2.selectbox("Tahun", YEAR, key="year")

        b1, b2 = st.columns([3, 1])
        submit = b1.form_submit_button("Mulai Klasifikasi", type="primary", width="stretch")
        b2.form_submit_button("Reset", on_click=reset_form, width="stretch")

    if submit:
        st.session_state.pop("pred", None)

        if (year, month) > (date.today().year, date.today().month):
            st.error("Periode yang dipilih belum terjadi.")
            st.stop()
        if (year, month) < (2013, 3):
            st.error("Citra Landsat 8 baru tersedia sejak Maret 2013.")
            st.stop()

        start_date = date(year, month, 1)
        end_date = date(year, month, calendar.monthrange(year, month)[1])

        with result_box.container():
            st.skeleton(400)

        try:
            with st.status("Memulai klasifikasi...") as status:
                status.update(label="Mengambil citra Landsat...")
                saved = save_landsat_images(ALL_SUBDISTRICT, start_date, end_date)

                if not saved:
                    status.update(label="Tidak ada citra Landsat.", state="error")
                    st.warning(f"Tidak ada citra untuk {MONTH[month - 1]} {year}.")
                    result_box.empty()
                    st.stop()

                boundary = get_boundary_file(str(BOUNDARY_PATH), ALL_SUBDISTRICT)

                status.update(label="Memproses citra (1/3)...")
                hasil = clipping_jakarta_boundary(boundary, file_name=saved, is_gee=True)
                hasil = cloud_masking(hasil)

                status.update(label="Memproses citra (2/3)...")
                hasil = rescalling_ST_SR(hasil)
                hasil = index_spectral(hasil)
                hasil = composite_image(hasil)
                status.update(label="Memproses citra (3/3)...")
                hasil = inverse_distance_weighting(hasil, boundary, n=8, r=10, p=2)

                status.update(label="Menghitung statistik dan prediksi...")
                hasil = count_statistic(hasil, boundary)
                pred = predict_uhi(hasil)

                if subdistrict != ALL_SUBDISTRICT:
                    pred = pred[pred["subdistrict"] == subdistrict]

                if pred.empty:
                    raise ValueError(
                        f"Tidak ada data valid untuk {subdistrict} pada "
                        f"{MONTH[month - 1]} {year} (kemungkinan tertutup awan)."
                    )

                data = {
                    "user_id": user["id"],
                    "subdistrict": subdistrict,
                    "year": year,
                    "month": month,
                    "total_image": len(saved),
                    "statistic": json.loads(pred.to_json(orient="records")),
                    "result": pred["kelas_uhi"].mode()[0],
                }

                create_history(data)

                st.session_state["pred"] = pred
                status.update(label="Klasifikasi selesai.", state="complete")

        except Exception as error:
            result_box.empty()
            st.error(f"Klasifikasi gagal: {error}")

if "pred" in st.session_state:
    with result_box.container():
        pred = st.session_state["pred"]

        sub = st.selectbox("Kecamatan", pred["subdistrict"].unique(), key="res_sub")
        row = pred[pred["subdistrict"] == sub].iloc[0]

        st.caption(f"Periode: {row['period']}")
        st.subheader(row["kelas_uhi"])

        for name in CLASS_NAMES:
            value = float(row[f"prob_{name.lower()}"])
            st.progress(value, text=f"{name}: {value:.0%}")

        if st.button("Lihat di peta"):
            st.switch_page("pages/uhi_mapping.py")