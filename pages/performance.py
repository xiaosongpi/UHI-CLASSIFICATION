import time
from pathlib import Path
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
EVAL_PATH = BASE_DIR / "src" / "preprocessing" / "models" / "evaluasi_model.json"

with open(EVAL_PATH, "r", encoding="utf-8") as file:
    metrics = json.load(file)

st.title("Evaluasi Model")
col_1, col_2 = st.columns(2)
with st.container(gap = "xlarge"):
    with col_1:
        labels = ["Rendah", "Normal", "Tinggi"]

        fig, ax = plt.subplots()
        sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set(xlabel="Prediksi", ylabel="Aktual")
        st.pyplot(fig)
        plt.close(fig)

    with col_2:
        per_class = metrics["per_class"]
        macro_precision = sum(v["precision"] for v in per_class.values()) / len(per_class)
        macro_recall = sum(v["recall"] for v in per_class.values()) / len(per_class)

        col_1, col_2 = st.columns(2)
        with col_1:
            with st.container(border=True):
                st.metric("Accuracy", f"{metrics['accuracy']:.2%}")

        with col_2:
            with st.container(border=True):
                st.metric("Macro Precision", f"{macro_precision:.2%}")

        col_1, col_2 = st.columns(2)
        with col_1:
            with st.container(border=True):
                st.metric("Macro Recall", f"{macro_recall:.2%}")

        with col_2:
            with st.container(border=True):
                st.metric("Macro F1-Score", f"{metrics['macro_f1']:.2%}")

            kelas_terbaik = ""
            kelas_terlemah = ""
            f1_terbaik = 0
            f1_terlemah = 1

            for nama, nilai in per_class.items():
                if nilai["f1_score"] > f1_terbaik:
                    f1_terbaik = nilai["f1_score"]
                    kelas_terbaik = nama
                if nilai["f1_score"] < f1_terlemah:
                    f1_terlemah = nilai["f1_score"]
                    kelas_terlemah = nama

        st.write(
            f"Model mencapai accuracy **{metrics['accuracy']:.2%}** pada **{metrics['n_test']} data uji**. "
            f"Kelas **{kelas_terbaik}** paling baik dikenali (F1-Score **{f1_terbaik:.2%}**), "
            f"sedangkan kelas **{kelas_terlemah}** paling lemah (F1-Score **{f1_terlemah:.2%}**). "
            f"Karena itu, hasil prediksi kelas **{kelas_terlemah}** perlu digunakan secara hati-hati."
        )