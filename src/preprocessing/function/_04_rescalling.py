import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def rescalling_ST_SR(images, is_visual = False):
    i = 0
    for period, group_image in images.items():
        for items in group_image:
            i += 1
            image = items['image']

            sr_b2 = image[0]
            sr_b3 = image[1]
            sr_b4 = image[2]
            sr_b5 = image[3]
            sr_b6 = image[4]
            st_b10 = image[5]

            for sr in (sr_b2, sr_b3, sr_b4, sr_b5, sr_b6):
                valid = sr != -9999
                sr[valid] = sr[valid] * 0.0000275 - 0.2

            valid = st_b10 != -9999
            st_b10[valid] = st_b10[valid] * 0.00341802 + 149 - 273.15

            if is_visual:
                Path("visual/rescaling").mkdir(parents = True, exist_ok = True)
 
                rgb = np.stack(image[[2, 1, 0]], axis = -1)
                valid = (rgb != -9999).all(axis = -1)
                rgb = np.clip(rgb / 0.3, 0, 1)
                rgb[~valid] = 0
                plt.imsave(f"visual/rescaling/{period}_{i}.png", rgb)

    return images