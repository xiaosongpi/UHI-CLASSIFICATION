import pandas as pd
from pathlib import Path
from function._01_boundary import get_boundary_file
from function._02_clipping import clipping_jakarta_boundary
from function._03_cloud_mask import cloud_masking
from function._04_rescalling import rescalling_ST_SR
from preprocessing.function._06_indeks_spektral import index_spectral
from function._07_composit_monthly import composite_image
from preprocessing.function._05_inverse_distance_weighting import inverse_distance_weighting, test_idw
from preprocessing.function._08_statistic_count import count_statistic
from preprocessing.function._09_labeling import natural_break_labeling
from preprocessing.function._10_train import train_final_model

BOUNDARY_PATH = "../../data/raw_data/boundary.gpkg"
TIF_PATH = "../../data/raw_data/landsat"

def run_preprocessing():
    print("1. MEMUAT BATAS ADMINISTRASI")
    print("=" * 60)
    boundary = get_boundary_file(BOUNDARY_PATH)

    print("2. CLIPPING CITRA LANDSAT")
    print("=" * 60)
    images = clipping_jakarta_boundary(boundary, TIF_PATH)

    print("3. CLOUD MASKING")
    print("=" * 60)
    images = cloud_masking(images)

    print("4. RESCALING BAND")
    print("=" * 60)
    images = rescalling_ST_SR(images)

    print("5. PERHITUNGAN INDEKS SPEKTRAL")
    print("=" * 60)
    images = index_spectral(images)

    print("6. DETEKSI OUTLIER")
    print("=" * 60)
    images = outlier_detection(images)

    print("7. KOMPOSIT BULANAN")
    print("=" * 60)
    images = composite_image(images)

    print("8. PENGISIAN GAP DENGAN IDW")
    print("=" * 60)
    n, r, p, results = test_idw(images)

    print("8. PENGISIAN GAP DENGAN IDW")
    print("=" * 60)
    images = inverse_distance_weighting(images, n, r, p)

    print("9. PERHITUNGAN STATISTIK KECAMATAN")
    print("=" * 60)
    dataframe = count_statistic(images, boundary)

    print("10. PELABELAN KELAS LST")
    print("=" * 60)
    dataframe = natural_break_labeling(dataframe)

    print("\nPreprocessing selesai.")
    print(f"Jumlah data: {len(dataframe)}")

    return dataframe

DATASET_PATH = Path("data/processed_dataset.csv")

def main():
    dataframe = run_preprocessing()
    dataframe = dataframe.reset_index(drop=True)

    DATASET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataframe.to_csv(
        DATASET_PATH,
        index=False
    )

    print(f"\nDataset tersimpan: {DATASET_PATH}")

    print("11. PELATIHAN MODEL")
    print("=" * 60)
    model, best_params, metrics, cv_df = train_final_model(dataframe)

    print("\n" + "=" * 60)
    print("PROSES KLASIFIKASI SELESAI")
    print(f"Parameter terbaik : {best_params}")
    print(f"Accuracy (uji)    : {metrics['accuracy']:.4f}")
    print(f"Macro F1 (uji)    : {metrics['f1_score']:.4f}")
    print("Output tersimpan di folder: models/")
    print("=" * 60)


if __name__ == "__main__":
    main()