import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def index_spectral(images):
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

    return images