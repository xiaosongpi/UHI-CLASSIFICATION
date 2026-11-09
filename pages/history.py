import logging

import pandas as pd
import streamlit as st

from src.services.historry_services import (
    get_history_by_user_id,
    delete_history
)

user = st.session_state.get("user")

if not user:
    if not st.session_state.pop("login_redirected", False):
        st.session_state["page"] = "login"
        st.session_state["open_dialog"] = True
        st.session_state["login_redirected"] = True
        st.rerun()
    st.stop()

st.title("Riwayat")

logger = logging.getLogger(__name__)
INTERNAL_ERROR = "Internal Server Error"

user_id = user["id"]

try:
    histories = get_history_by_user_id(user_id)
except Exception as error:
    logger.error("Gagal mengambil riwayat", exc_info=error)
    st.error(INTERNAL_ERROR)
    st.stop()

if histories:
    df = pd.DataFrame(histories)

    df = df.drop(columns=["user_id", "id", "statistic"], errors="ignore")

    if "created_at" in df.columns:
        df["created_at"] = (
            pd.to_datetime(df["created_at"], errors="coerce")
            .dt.strftime("%d-%m-%Y %H:%M")
            .fillna("-")
        )

    event = st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=500,
        on_select="rerun",
        selection_mode="multi-row"
    )

    selected_rows = event.selection.rows

    if selected_rows:
        if st.button("Hapus Riwayat Terpilih", type="secondary"):
            try:
                for index in selected_rows:
                    delete_history(histories[index]["id"])
            except Exception as error:
                logger.error("Gagal menghapus riwayat", exc_info=error)
                st.error(INTERNAL_ERROR)
            else:
                st.rerun()

    df = df.drop(columns=["user_id", "id"], errors="ignore")

    st.download_button(
        "Download CSV",
        data = df.to_csv(index=False).encode("utf-8"),
        file_name = "riwayat.csv",
        mime = "text/csv"
    )

else:
    st.info("Belum ada riwayat.")