import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def composite_image(images, is_visual = False):
    i = 0
    for period, group_image in images.items():
        i += 1
        image_list = []

        for items in group_image:
            image = items['image']

            image[image == -9999] = np.nan

            image_list.append(image)

        image_list = np.stack(image_list, axis = 0)
        median_image = np.nanmedian(image_list, axis = 0)
        median_image[np.isnan(median_image)] = -9999

        group_image[:] = [group_image[0]]
        group_image[0]['image'] = median_image

        if is_visual:
            Path("visual/composite").mkdir(parents = True, exist_ok = True)
 
            nama_band = ["LST", "NDVI", "NDBI", "MNDWI"]
            idx_band = [5, 6, 7, 8]
            nama_cmap = ["YlOrRd", "Greens", "Reds", "Blues"]
            batas_bawah = [20, -1, -1, -1]
            batas_atas = [45, 1, 1, 1]
 
            for k in range(4):
                data = np.ma.masked_equal(median_image[idx_band[k]], -9999)
                cmap = plt.get_cmap(nama_cmap[k]).copy()
                cmap.set_bad("black")
 
                plt.figure(figsize = (6, 6))
                plt.imshow(data, cmap = cmap, vmin = batas_bawah[k], vmax = batas_atas[k])
                plt.colorbar(label = nama_band[k], shrink = 0.8)
                plt.title(f"{nama_band[k]} {period}")
                plt.axis("off")
                plt.savefig(f"visual/composite/{nama_band[k]}_{period}_{i}.png", dpi = 150, bbox_inches = "tight")
                plt.close()

    return images