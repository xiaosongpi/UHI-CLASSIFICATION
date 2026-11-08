import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def index_spectral(images, is_visual = None):
    i = 0
    for period, group_image in images.items():
        for items in group_image:
            i += 1
            image = items['image']

            sr_b3 = image[1]
            sr_b4 = image[2]
            sr_b5 = image[3]
            sr_b6 = image[4]
            result = []

            for band_1, band_2 in ((sr_b5, sr_b4), (sr_b6, sr_b5), (sr_b3, sr_b6)):
                index_spectral = np.full(band_1.shape, -9999, dtype = "float32")
                valid = (band_1 != -9999) & (band_2 != -9999)
                index_spectral[valid] = (band_1[valid] - band_2[valid]) / (band_1[valid] + band_2[valid]) 

                invalid = valid & ~((index_spectral >= -1) & (index_spectral <= 1))
                index_spectral[invalid] = -9999
                
                result.append(index_spectral)

            items['image'] = np.concatenate(
                [image[:6], np.stack(result, axis=0)], axis=0
            )

            if is_visual:
                Path("visual/index_spectral").mkdir(parents = True, exist_ok = True)
 
                nama_index = ["NDVI", "NDBI", "MNDWI"]
                nama_cmap = ["Greens", "Reds", "Blues"]
 
                for k in range(3):
                    data = np.ma.masked_equal(result[k], -9999)
                    cmap = plt.get_cmap(nama_cmap[k]).copy()
                    cmap.set_bad("black")
 
                    plt.figure(figsize = (6, 6))
                    plt.imshow(data, cmap = cmap, vmin = -1, vmax = 1)
                    plt.colorbar(label = nama_index[k], shrink = 0.8)
                    plt.title(f"{nama_index[k]} {period}")
                    plt.axis("off")
                    plt.savefig(f"visual/index_spectral/{nama_index[k]}_{period}_{i}.png", dpi = 150, bbox_inches = "tight")
                    plt.close()
        
    return images