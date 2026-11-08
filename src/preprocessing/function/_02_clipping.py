import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path.cwd()
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from rasterio.io import MemoryFile
from rasterio.mask import mask as rasterio_mask
from src.services.gee_services import get_supabase_file

def clipping_jakarta_boundary(boundary, file_path_tif = None, file_name = None, is_gee = False, is_visual = False):    
    if is_gee:
        image_source = file_name
        band = None
    else:
        image_source = sorted(Path(file_path_tif).glob("*.tif"))
        band = [2, 3, 4, 5, 6, 9, 18]
    
    image = {}
    for item in image_source:
        if is_gee:
            name = item
            data = get_supabase_file(item)
        else:
            name = item.name
            data = item.read_bytes()

        with MemoryFile(data) as file:
            with file.open() as src:
                mask_image, transform = rasterio_mask(
                    src, 
                    (boundary.to_crs(src.crs).geometry),
                    crop = True,
                    filled = False,
                    indexes = band,
                )

                mask_image = mask_image.astype("float32").filled(-9999)

                period = name.split("_")[1][:7]

                if period not in image:
                    image[period] = []

                image[period].append({
                    "image": mask_image,
                    "mask_image": (mask_image != -9999).all(axis = 0),
                    "transform": transform,
                    "crs": src.crs,
                })

                if is_visual:
                    Path("visual/raw").mkdir(parents = True, exist_ok = True)
                    Path("visual/clipping").mkdir(parents = True, exist_ok = True)
                    nama_file = Path(name).stem

                    raw = np.stack(src.read([4, 3, 2]), axis = -1)
                    raw = np.nan_to_num(raw)
                    raw = raw * 0.0000275 - 0.2
                    raw = np.clip(raw / 0.3, 0, 1)
                    plt.imsave(f"visual/raw/{nama_file}.png", raw)

                    clip = np.stack(mask_image[[2, 1, 0]], axis = -1)
                    clip = np.nan_to_num(clip)
                    valid = (clip != -9999).all(axis = -1)
                    clip = clip * 0.0000275 - 0.2
                    clip = np.clip(clip / 0.3, 0, 1)
                    clip[~valid] = 0
                    plt.imsave(f"visual/clipping/{nama_file}.png", clip)

    return image
