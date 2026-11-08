import streamlit as st

st.markdown(
    """
        <style>
            [data-testid="stMain"] [data-testid="stMainBlockContainer"] {
                display: flex;
                flex-direction: column;
                width: 100%;
                max-width: 1200px;
            }
        </style>
    """,
    unsafe_allow_html=True,
)

with st.container(gap="xlarge"):
    with st.container(border=True, height = 500, vertical_alignment="center"):
        col1, col2 = st.columns(2)

        with col1:
            st.title("JakUHI: Klasifikasi Tingkat UHI")
            st.write(
                "Sistem klasifikasi tingkat Urban Heat Island (UHI) "
                "berbasis citra satelit Landsat 8 Collection 2 Level 2 pada 42 kecamatan "
                "di DKI Jakarta."
            )
            if st.button("Mulai klasifikasi", type = "primary"):
                st.switch_page("pages/uhi_classification.py")

        with col2:
            st.image("images/Indonesian.png")

    with st.container():
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            with st.container(border=True, height = 300):
                with st.container(height = 200, border = False):
                    st.badge("UHI Klasifikasi", color="green")
                    st.markdown("##### Klasifikasi UHI")
                    st.write(
                        "Klasifikasikan tingkat UHI berdasarkan "
                        "kecamatan dan periode pengamatan."
                    )
                if st.button("Klik sini", type = "tertiary", key = "btn-1"):
                    st.switch_page("pages/uhi_classification.py")

        with col2:
            with st.container(border=True, height = 300):
                with st.container(height = 200, border = False):
                    st.badge("Peta UHI", color="yellow")
                    st.markdown("##### Peta UHI")
                    st.write(
                        "Lihat persebaran tingkat UHI pada "
                        "kecamatan di DKI Jakarta."
                    )
                if st.button("Klik sini", type = "tertiary", key = "btn-2"):
                    st.switch_page("pages/uhi_mapping.py")

        with col3:
            with st.container(border=True, height = 300):
                with st.container(height = 200, border = False):
                    st.badge("Performa Model", color="orange")
                    st.markdown("##### Evaluasi Model")
                    st.write(
                        "Lihat performa model melalui confusion matrix "
                        "dan metrik evaluasi."
                    )
                if st.button("Klik sini", type = "tertiary", key = "btn-3"):
                    st.switch_page("pages/performance.py")

        with col4:
            with st.container(border=True, height = 300):
                with st.container(height = 200, border = False):
                    st.badge("Riwayat", color="red")
                    st.markdown("##### Riwayat Klasifikasi")
                    st.write(
                        "Lihat kembali riwayat klasifikasi "
                        "yang telah dilakukan."
                    )
                if st.button("Klik sini", type = "tertiary", key = "btn-4"):
                    st.switch_page("pages/history.py")

    with st.container(gap = "xlarge"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("# Tujuan Sistem")

        with col2:
            st.write(
                "JakUHI bertujuan menyediakan informasi "
                "klasifikasi dan persebaran tingkat UHI di "
                "DKI Jakarta sebagai sarana untuk memahami "
                "fenomena pulau panas perkotaan."
            )

    with st.container():
        st.markdown("# Cara Kerja", text_alignment = "center")
        with st.container(border = False, horizontal_alignment = "center", vertical_alignment = "center"):
            col1, col2 = st.columns([1, 10], vertical_alignment="center")
            with col1:
                st.markdown("## 1.", text_alignment="center")
            with col2:
                with st.container(border=True):
                    st.markdown("##### Mulai klasifikasi tingkat UHI")
                    st.caption("Masuk halaman klasifikasi UHI untuk melakukan mulai klasifikasi. Proses dimulai dari pemilihan kecamatan, tanggal, dan bulan yang diindinkan")
        
        with st.container(border = False, horizontal_alignment = "center", vertical_alignment = "center"):
            col1, col2 = st.columns([1, 10], vertical_alignment="center")
            with col1:
                st.markdown("## 2.", text_alignment="center")
            with col2:
                with st.container(border=True):
                    st.markdown("##### Melihat persebaran tingkat UHI di 42 kecamatan DKI Jakarta")
                    st.caption("Setelah melakukan proses klasifikasi UHI, masuk halaman peta UHI untuk melihat persebaran tingkat UHI di 42 Kecamatan DKI Jakarta")


    with st.container(border = True, vertical_alignment="center", horizontal_alignment="center", height=300):
        st.markdown("# Eksplorasi lebih luas", text_alignment = "center")
        st.markdown("##### Ingin memahmi lebih dalam terkait cara penggunaan sistem klasifikasi tingkat UHI", text_alignment = "center")
        if st.button("Klik sini", type = "primary"):
            st.switch_page("pages/help.py")