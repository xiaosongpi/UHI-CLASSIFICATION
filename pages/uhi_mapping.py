import streamlit as st
import geopandas as gpd
import plotly.express as px
from pathlib import Path
import re
import pandas as pd

from src.preprocessing.function._01_boundary import get_boundary_file
from src.services.historry_services import get_history_by_user_id

BASE_DIR = Path(__file__).resolve().parents[1]
BOUNDARY_PATH = BASE_DIR / "data" / "raw_data" / "boundary.gpkg"

INDEX_COLS = {
    "ndvi_mean": "NDVI mean",
    "ndvi_std": "NDVI std",
    "ndvi_min": "NDVI min",
    "ndvi_max": "NDVI max",
    "ndvi_q1": "NDVI q1",
    "ndvi_q3": "NDVI q3",
    "ndvi_median": "NDVI median",

    "ndbi_mean": "NDBI mean",
    "ndbi_std": "NDBI std",
    "ndbi_min": "NDBI min",
    "ndbi_max": "NDBI max",
    "ndbi_q1": "NDBI q1",
    "ndbi_q3": "NDBI q3",
    "ndbi_median": "NDBI median",

    "mndwi_mean": "MNDWI mean",
    "mndwi_std": "MNDWI std",
    "mndwi_min": "MNDWI min",
    "mndwi_max": "MNDWI max",
    "mndwi_q1": "MNDWI q1",
    "mndwi_q3": "MNDWI q3",
    "mndwi_median": "MNDWI median",
}

LABEL_MAP = {
    "RENDAH": "Rendah",
    "NORMAL": "Normal",
    "TINGGI": "Tinggi",
}
KNOWN_LABELS = list(LABEL_MAP.values())

COLOR_MAP = {
    "Rendah": "green",
    "Normal": "yellow",
    "Tinggi": "red",
    "Belum diklasifikasikan": "lightgray",
}

MONTH_NAMES = {
    1: "Januari", 
    2: "Februari", 
    3: "Maret", 
    4: "April",
    5: "Mei", 
    6: "Juni", 
    7: "Juli", 
    8: "Agustus",
    9: "September", 
    10: "Oktober", 
    11: "November", 
    12: "Desember",
}


def month_label(value):
    try:
        return MONTH_NAMES[int(value)]
    except (TypeError, ValueError, KeyError):
        return str(value)


def year_label(value):
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def norm(value):
    """Normalisasi nama kecamatan: huruf besar, tanpa 'Kec.', tanpa simbol."""
    text = re.sub(r"\b(KECAMATAN|KEC)\b\.?", " ", str(value).upper())
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return " ".join(text.split())


def normalize_label(value):
    text = str(value).strip().upper()
    if text in LABEL_MAP:
        return LABEL_MAP[text]
    for key, label in LABEL_MAP.items():
        if key in text:
            return label
    return str(value).strip()


def history_label(item):
    return (
        f"{year_label(item.get('year'))} - "
        f"{month_label(item.get('month'))} - "
        f"{item.get('subdistrict')}"
    )


def history_to_rows(item):
    """Ubah satu item history menjadi daftar hasil per kecamatan.

    Hasil per kecamatan ada di item["statistic"] (list of dict), masing-masing
    punya "subdistrict" dan "kelas_uhi" sendiri.
    """
    statistic = item.get("statistic") or []

    if isinstance(statistic, dict): 
        statistic = [statistic]

    rows = []
    for row in statistic:
        if not isinstance(row, dict):
            continue
        rows.append({
            "subdistrict": row.get("subdistrict", item.get("subdistrict")),
            "kelas_uhi": row.get("kelas_uhi", item.get("result")),
            **{col: row.get(col) for col in INDEX_COLS},
        })

    return rows


def load_boundary():
    boundary = get_boundary_file(
        str(BOUNDARY_PATH),
        "Seluruh kecamatan"
    )
    return boundary.to_crs(epsg=4326)


st.title("Peta UHI")

boundary = load_boundary()
boundary["WADMKC"] = boundary["WADMKC"].astype(str).str.strip()
boundary["key"] = boundary["WADMKC"].map(norm)

user_id = st.session_state["user"]["id"]
history = get_history_by_user_id(user_id)

has_history = bool(history)

if has_history:
    options = {}
    for item in sorted(history, key=lambda h: str(h.get("created_at")), reverse=True):
        options.setdefault(history_label(item), item)

    selected = st.selectbox("Riwayat pencarian", list(options.keys()))
    selected_item = options[selected]

    data = pd.DataFrame(
        history_to_rows(selected_item),
        columns=["subdistrict", "kelas_uhi", *INDEX_COLS],
    )

    data["key"] = data["subdistrict"].map(norm)
    data["kelas_uhi"] = data["kelas_uhi"].map(normalize_label)
    for col in INDEX_COLS:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.drop_duplicates("key", keep="last")

    boundary = boundary.merge(
        data[["key", "kelas_uhi", *INDEX_COLS]],
        on="key",
        how="left"
    )

    boundary["kelas_uhi"] = boundary["kelas_uhi"].fillna(
        "Belum diklasifikasikan"
    )

else:
    boundary["kelas_uhi"] = "Belum diklasifikasikan"

    for col in INDEX_COLS:
        boundary[col] = None


with st.container(border=True):
    f_col1, f_col2 = st.columns(2)

    with f_col1:
        wilayah = st.selectbox(
            "Batas administratif",
            ["Seluruh kecamatan"] + sorted(
                boundary["WADMKK"].dropna().unique()
            )
        )

    with f_col2:
        tingkat = st.pills(
            "Tingkat UHI",
            KNOWN_LABELS,
            selection_mode="multi",
            default=KNOWN_LABELS
        )


filtered = boundary.copy()

if wilayah != "Seluruh kecamatan":
    filtered = filtered[filtered["WADMKK"] == wilayah]

if has_history:
    filtered = filtered[
        ~filtered["kelas_uhi"].isin(KNOWN_LABELS)
        | filtered["kelas_uhi"].isin(list(tingkat or []))
    ]


map_col, list_col = st.columns(2)

with map_col:
    fig = px.choropleth_map(
        filtered,
        geojson=filtered.__geo_interface__,
        locations="WADMKC",
        featureidkey="properties.WADMKC",
        color="kelas_uhi",
        color_discrete_map=COLOR_MAP,
        hover_name="WADMKC",
        hover_data={
            "kelas_uhi": True,
            **{col: ":.3f" for col in INDEX_COLS}
        },
        labels=INDEX_COLS,
        center={"lat": -6.2, "lon": 106.86},
        zoom=9.5,
        map_style="carto-positron",
        height=500
    )

    fig.update_traces(
        marker_line_color="black",
        marker_line_width=0.5
    )

    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        legend={"orientation": "h"}
    )

    st.plotly_chart(fig, width="stretch")


with list_col:
    with st.container(height=500):
        st.markdown("#### Daftar kecamatan")

        for _, row in filtered.iterrows():
            c_name, c_badge = st.columns(
                [3, 2],
                vertical_alignment="center"
            )

            with c_name:
                st.write(row["WADMKC"])

            with c_badge:
                st.badge(
                    row["kelas_uhi"],
                    color=COLOR_MAP.get(row["kelas_uhi"], "gray")
                )