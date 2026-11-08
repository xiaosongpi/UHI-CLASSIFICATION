import streamlit as st

st.title("Bantuan")

if "help" not in st.session_state:
    st.session_state.help = 1

def next_page():
    st.session_state.help += 1

col1, col2 = st.columns([1, 2], gap = "xxlarge")
with col1:
    with st.container():

        if st.button("Cara menggunakan sistem klasifikasi UHI", width="stretch"):
            st.session_state.help = 1

        if st.button("Panduan klasifikasi UHI", width="stretch"):
            st.session_state.help = 2

        if st.button("Panduan peta UHI", width="stretch"):
            st.session_state.help = 3

        if st.button("Evaluasi model", width="stretch"):
            st.session_state.help = 4

        if st.button("Riwayat klasifikasi", width="stretch"):
            st.session_state.help = 5

        if st.button("Pengertian UHI dan LST", width="stretch"):
            st.session_state.help = 6

        if st.button("Apa itu NDVI, NDBI, dan MNDWI", width="stretch"):
            st.session_state.help = 7

        if st.button("Arti kelas Rendah, Normal, dan Tinggi", width="stretch"):
            st.session_state.help = 8

        if st.button("Model yang digunakan", width="stretch"):
            st.session_state.help = 9

with col2:
    if st.session_state.help == 1:
        st.subheader("Cara menggunakan sistem klasifikasi UHI")
        st.write("Untuk menggunakan sistem klasifikasi UHI diperlukan beberapa tahap yang perlu dilalui. Berikut merupakan rincian tahapan yang perlu dilakukan sebelum dapat mengklasifikasi tingkat UHI")
        st.markdown("""
            - Login: tahap awal sebelum melakukan klasifikasi UHI agar seluruh klasifikasi yang dilakukan dapat terseimpan
            - Mulai klasifikasi dengan mengakses halaman Klasifikasi UHI: Setelah melakukan login, anda dapat memilih kecamatan yang ingin diklasifikasi. Pada pemilihan ini anda dapat memilih satu kecamatan saja dan dapat memilih seluruh kecamatan.
            - Mengunjungi halaman Peta UHI: Untuk melihat hasill klasifikasi pada peta, dapat mengunjungi halaman Peta UHI untuk melihat bagaimana persebaran tingkat UHI di 42 kecamatan DKI Jakarta.
        """)
        st.button("lanjut", type = "tertiary", on_click = next_page)

    if st.session_state.help == 2:
        st.subheader("Panduan klasifikasi UHI")
        st.write("Untuk mengklasifikasi UHI, terdapat beberapa pilihan yang perlu anda iskan")
        st.markdown(
            """
                - Kecamatan: Opsi ini memungkinkan anda untuk mengklasifikasi tingkat UHI berdasarkan kecamatan yang anda pilih attau seluruh kecamatan
                - Tahun dan Bulan:  Masukkan ini memungkinkan anda untuk mengambil citra landsat 8 Collection 2 Level 2 berdasarkan Tahun dan Bulan yang anda tentukan. 
            """
        )
        st.write("Perlu diperhatikan: ")
        st.markdown("""
                - Citra Lnadsat 8 Collection 2 Level 2 hanya tersedia data yang dimulai dari tahun 2013.
            """
        )
        st.button("lanjut", type = "tertiary", on_click = next_page)

    if st.session_state.help == 3:
        st.subheader("Panduan peta UHI")
        st.write(
            "Halaman Peta UHI menampilkan sebaran tingkat UHI di 42 kecamatan DKI Jakarta "
            "berdasarkan hasil klasifikasi yang pernah Anda lakukan. Berikut cara membacanya."
        )

        st.markdown("""
            - Riwayat pencarian: Pilih salah satu riwayat klasifikasi (format: Tahun - Bulan - Kecamatan) untuk ditampilkan pada peta. Riwayat diurutkan dari yang terbaru.
            - Batas administratif: Pilih "Seluruh kecamatan" untuk melihat semua wilayah, atau pilih salah satu kota administratif.
            - Tingkat UHI: Pilih satu atau lebih tingkat (Rendah, Normal, Tinggi) untuk menyaring kecamatan yang ditampilkan.
        """)

        st.markdown("Cara membaca peta")
        st.markdown("""
            - Setiap kecamatan diberi warna sesuai tingkat UHI yaitu
                - Hijau : kelas Rendah
                - Kuning : kelas Normal
                - Merah : kelas Tinggi
            - Arahkan kursor ke sebuah kecamatan untuk melihat nama kecamatan, tingkat UHI, serta nilai statistik indeks (NDVI, NDBI, dan MNDWI).
            - Peta dapat digeser dan diperbesar atau diperkecil sesuai kebutuhan.
        """)

        st.markdown("Perlu diperhatikan")
        st.markdown("Halaman ini akan tertampilkan ketika klaifikasi sudah pernah dilakukan pada halaman Klasifikasi UHI")

        st.button("lanjut", type = "tertiary", on_click = next_page)
    if st.session_state.help == 4:
        st.subheader("Evaluasi model")
        st.write(
            "Halaman Evaluasi model menunjukkan seberapa baik model menebak tingkat UHI. Semakin tinggi nilainya, semakin dapat diandalkan hasil klasifikasi sistem. Model yang telah dilatih kemudian akan dites dengan data yang belum pernah digunakan.")

        st.markdown("""
            - **Akurasi**: dari seluruh soal ujian, berapa persen jawaban model yang benar. Misalnya akurasi 70% berarti dari 100 kecamatan, sekitar 70 ditebak dengan benar.
            - **Presisi**: ketika model menjawab suatu kelas (misalnya Tinggi), seberapa sering jawabannya benar. Presisi tinggi berarti model jarang salah menuduh.
            - **Recall**: dari seluruh kecamatan yang sebenarnya berkelas tertentu (misalnya Tinggi), berapa persen yang berhasil ditemukan model. Recall tinggi berarti model jarang melewatkan.
            - **F1-score**: nilai gabungan presisi dan recall dalam satu angka. Semakin mendekati 100%, semakin seimbang dan baik.
        """)

        st.button("lanjut", type = "tertiary", on_click = next_page)
    if st.session_state.help == 5:
        st.subheader("Riwayat klasifikasi")
        st.write("Halaman Riwayat menyimpan seluruh klasifikasi UHI yang pernah Anda lakukan, hasilnya dapat diunduh kedalam format CSV.")

        st.markdown("**Cara menghapus riwayat**")
        st.markdown("""
            - Pilih satu atau beberapa baris pada tabel dengan mencentang baris yang diinginkan.
            - Setelah ada baris yang dipilih, tombol **Hapus Riwayat Terpilih** akan muncul. Klik tombol tersebut untuk menghapus riwayat yang dipilih.
        """)

        st.write("Perlu diperhatikan: ")
        st.markdown("""
            - Riwayat yang sudah dihapus tidak dapat dikembalikan.
            - Jika belum pernah melakukan klasifikasi, halaman akan menampilkan pesan "Belum ada riwayat".
        """)

        st.button("lanjut", type = "tertiary", on_click = next_page)
    if st.session_state.help == 6:
        st.subheader("Pengertian UHI dan LST")
        st.write("UHI dan LST adalah dua istilah utama yang digunakan dalam sistem ini. Keduanya saling berkaitan: LST menjadi ukuran suhu yang digunakan untuk memahami UHI.")

        st.markdown("**Apa itu UHI (Urban Heat Island)?**")
        st.write("Urban Heat Island (UHI) atau pulau panas perkotaan adalah fenomena ketika suatu kawasan perkotaan memiliki suhu yang lebih tinggi dibandingkan wilayah di sekitarnya. Fenomena ini umum terjadi di kota besar seperti DKI Jakarta.")
        st.markdown("""
            Beberapa penyebab terjadinya UHI:
            - Banyaknya permukaan yang menyerap dan menyimpan panas, seperti beton, aspal, dan atap bangunan.
            - Berkurangnya vegetasi dan ruang terbuka hijau yang berfungsi menyejukkan lingkungan.
            - Panas yang dihasilkan dari aktivitas manusia, seperti kendaraan, pendingin ruangan, dan industri.
        """)

        st.markdown("Apa itu LST (Land Surface Temperature)?")
        st.write("Land Surface Temperature (LST) atau suhu permukaan tanah adalah suhu pada permukaan bumi, seperti tanah, bangunan, jalan, dan vegetasi, yang diukur oleh sensor satelit melalui pita termal (thermal band). Pada sistem ini, LST diperoleh dari citra Landsat 8.")

        st.markdown("**Hubungan LST dan UHI**")
        st.write("LST digunakan sebagai indikator untuk mengamati UHI. Kawasan dengan LST yang tinggi menunjukkan permukaan yang lebih panas, sehingga dapat dikaitkan dengan intensitas UHI yang lebih tinggi. Dengan citra satelit, pola suhu ini dapat dipantau pada seluruh wilayah secara serentak, termasuk di tiap kecamatan.")

        st.button("lanjut", type = "tertiary", on_click = next_page)
    if st.session_state.help == 7:
        st.subheader("Apa itu NDVI, NDBI, dan MNDWI")
        st.write(
            "NDVI, NDBI, dan MNDWI adalah indeks spektral, yaitu nilai hasil perhitungan dari beberapa "
            "pita (band) citra satelit untuk menonjolkan jenis permukaan tertentu. "
            "Pada sistem ini, ketiganya menjadi fitur yang digunakan model untuk mengklasifikasi tingkat UHI. "
            "Nilai ketiganya berkisar antara -1 sampai 1."
        )

        st.markdown("NDVI (Normalized Difference Vegetation Index)")
        st.write("NDVI menunjukkan tingkat kerapatan dan kesehatan vegetasi.")
        st.markdown("""
            - Nilai tinggi (mendekati 1): vegetasi lebat, seperti taman, hutan kota, dan area hijau.
            - Nilai rendah (mendekati 0): lahan terbuka, permukaan beraspal, atau bangunan.
            - Nilai negatif: umumnya air atau awan.
        """)

        st.markdown("NDBI (Normalized Difference Built-up Index)")
        st.write("NDBI menunjukkan tingkat lahan terbangun.")
        st.markdown("""
            - Nilai tinggi: lahan terbangun.
            - Nilai rendah: vegetasi atau air.
        """)

        st.markdown("**MNDWI (Modified Normalized Difference Water Index)**")
        st.write("MNDWI menunjukkan keberadaan permukaan air, seperti laut, sungai, waduk.")
        st.markdown("""
            - Nilai tinggi: kemungkinan besar permukaan air.
            - Nilai rendah: lahan terbangun atau vegetasi.
        """)

        st.write(
            "Nilai-nilai inilah yang tampil saat Anda mengarahkan kursor ke sebuah kecamatan di halaman Peta UHI."
        )

        st.button("lanjut", type = "tertiary", on_click = next_page)
    if st.session_state.help == 8:
        st.subheader("Arti kelas Rendah, Normal, dan Tinggi")
        st.write(
            "Hasil klasifikasi pada sistem ini dikelompokkan menjadi tiga kelas tingkat UHI, "
            "yaitu Rendah, Normal, dan Tinggi. Setiap kecamatan mendapat satu kelas untuk "
            "bulan dan tahun yang dipilih."
        )

        st.markdown("**Arti tiap kelas**")
        st.markdown("""
            - *Rendah: suhu permukaan (LST) kecamatan tergolong paling rendah dibandingkan wilayah lain di Jakarta. Umumnya didukung oleh vegetasi yang lebih banyak atau keberadaan badan air.
            - *Normal: suhu permukaan kecamatan berada pada tingkat menengah. Kelas ini menjadi pembanding bagi dua kelas lainnya.
            - Tinggi: suhu permukaan kecamatan tergolong paling tinggi dibandingkan wilayah lain di Jakarta. Umumnya didominasi lahan terbangun dan vegetasi yang sedikit.
        """)

        st.markdown("**Bagaimana kelas ditentukan**")
        st.write(
            "Kelas dibentuk dari nilai LST pada citra Landsat 8. Nilai LST di seluruh wilayah "
            "Jakarta dikelompokkan menjadi tiga kelompok dengan metode natural breaks (Jenks), "
            "yaitu metode yang mencari batas alami antar kelompok nilai. Kelas sebuah kecamatan "
            "ditentukan dari kelas yang paling banyak muncul pada piksel-piksel di dalamnya."
        )

        st.button("lanjut", type = "tertiary", on_click = next_page)
    if st.session_state.help == 9:
        st.subheader("Model yang digunakan")
        st.write(
            "Sistem ini menggunakan model Feature Tokenizer Transformer (FT-Transformer), "
            "yaitu model kecerdasan buatan (deep learning) yang dirancang untuk data berbentuk tabel."
        )

        st.markdown("**Cara kerja secara singkat**")
        st.markdown("""
            - Model menerima nilai NDVI, NDBI, dan MNDWI dari kecamatan yang dipilih.
            - Model mempelajari pola hubungan antara nilai-nilai tersebut dan tingkat UHI dari data-data sebelumnya.
            - Hasilnya berupa kelas Rendah, Normal, atau Tinggi untuk tiap kecamatan.
        """)

        st.write("Perlu diperhatikan: ")
        st.markdown("""
            - Hasil adalah prediksi, bukan pengukuran suhu langsung, sehingga masih mungkin keliru.
            - Tingkat keandalan model dapat dilihat pada halaman **Evaluasi model**.
        """)

        st.button("lanjut", type = "tertiary", on_click = next_page)
    # if st.session_state.help == 10:
