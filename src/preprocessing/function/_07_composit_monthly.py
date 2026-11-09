import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def composite_image(images):
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

    return images