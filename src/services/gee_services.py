import ee
import requests

from src.database.client import get_supabase
from src.preprocessing.function._00_GEE import gee_init, get_landsat_image, get_region

gee_init()

BUCKET = "gee-collection"
BANDS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "ST_B10", "QA_PIXEL"]

def upload_file(file, file_name):
    return get_supabase().storage.from_(BUCKET).upload(
        path=file_name,
        file=file,
        file_options={
            "content-type": "image/tiff",
            "upsert": "true",
        },
    )

def save_landsat_images(area, start_date, end_date):
    region = get_region(area)
    image_ids = (
        get_landsat_image(area, start_date, end_date)
        .aggregate_array("system:id")
        .getInfo()
    )

    saved, errors = [], []

    for index, image_id in enumerate(image_ids):
        try:
            image = ee.Image(image_id).select(BANDS)

            url = image.getDownloadURL({
                "region": region,
                "scale": 30,
                "format": "GEO_TIFF",
            })

            response = requests.get(url, timeout=120)
            response.raise_for_status()

            date = image_id.split("_")[-1]
            file_name = (
                f"Landsat_{date[:4]}-{date[4:6]}-{date[6:]}"
                f"_{index}.tif"
            )

            upload_file(response.content, file_name)
            saved.append(file_name)

            print("OK", file_name)

        except Exception as e:
            print("GAGAL", image_id, e)
            errors.append(f"{image_id}: {e}")

    if image_ids and not saved:
        raise RuntimeError(f"{len(image_ids)} citra ditemukan, tapi semua gagal. Contoh error: {errors[0]}")

    return saved

def get_supabase_file(file_name):
    file = get_supabase().storage.from_("gee-collection").download(file_name)
    return file