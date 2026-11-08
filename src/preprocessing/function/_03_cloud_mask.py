import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def cloud_masking(images, is_visual = False):
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
            qa_pixel = image[6].astype(np.int32)

            mask_cloud = (
                ((qa_pixel & (1 << 0)) != 0) |
                ((qa_pixel & (1 << 1)) != 0) |
                ((qa_pixel & (1 << 2)) != 0) |
                ((qa_pixel & (1 << 3)) != 0) |
                ((qa_pixel & (1 << 4)) != 0)
            )

            sr_b2[mask_cloud] = -9999
            sr_b3[mask_cloud] = -9999
            sr_b4[mask_cloud] = -9999
            sr_b5[mask_cloud] = -9999
            sr_b6[mask_cloud] = -9999
            st_b10[mask_cloud] = -9999

            items['image'] = image[:6]

            if is_visual:
                Path("visual/cloud_masking").mkdir(parents = True, exist_ok = True)

                rgb = np.stack(image[[2, 1, 0]], axis = -1)
                valid = (rgb != -9999).all(axis = -1)
                rgb = rgb * 0.0000275 - 0.2
                rgb = np.clip(rgb / 0.3, 0, 1)
                rgb[~valid] = 0
                plt.imsave(f"visual/cloud_masking/{period}_{i}.png", rgb)

    return images

# import numpy as np
# from pathlib import Path
# import matplotlib.pyplot as plt
# from scipy.ndimage import binary_dilation

# def cloud_masking(images, is_visual = False):
#     i = 0 
#     for period, group_image in images.items():
#         for items in group_image:
#             i += 1
#             image = items['image']

#             sr_b2 = image[0]
#             sr_b3 = image[1]
#             sr_b4 = image[2]
#             sr_b5 = image[3]
#             sr_b6 = image[4]
#             st_b10 = image[5]
#             inside = image[6] != -9999                  # <- tambah: area dalam Jakarta
#             qa_pixel = image[6].astype(np.int32)

#             mask_cloud = (
#                 ((qa_pixel & (1 << 0)) != 0) |
#                 ((qa_pixel & (1 << 1)) != 0) |
#                 ((qa_pixel & (1 << 2)) != 0) |
#                 ((qa_pixel & (1 << 3)) != 0) |
#                 ((qa_pixel & (1 << 4)) != 0)
#             )

#             mask_cloud &= inside                        # <- tambah: buang efek -9999 di luar batas
#             mask_cloud = binary_dilation(mask_cloud, iterations = 3) & inside   # <- tambah: lebarkan mask

#             pct_awan = mask_cloud[inside].mean() * 100  # <- tambah: persen awan di Jakarta
#             items['pct_awan'] = pct_awan                # <- tambah: disimpan untuk filter_cloud
#             print(f"{period} | scene {i} | awan di Jakarta: {pct_awan:.1f}%")

#             sr_b2[mask_cloud] = -9999
#             sr_b3[mask_cloud] = -9999
#             sr_b4[mask_cloud] = -9999
#             sr_b5[mask_cloud] = -9999
#             sr_b6[mask_cloud] = -9999
#             st_b10[mask_cloud] = -9999

#             items['image'] = image[:6]

#             if is_visual:
#                 Path("visual/cloud_masking").mkdir(parents = True, exist_ok = True)

#                 rgb = np.stack(image[[2, 1, 0]], axis = -1)
#                 valid = (rgb != -9999).all(axis = -1)
#                 rgb = rgb * 0.0000275 - 0.2
#                 rgb = np.clip(rgb / 0.3, 0, 1)
#                 rgb[~valid] = 0
#                 plt.imsave(f"visual/cloud_masking/{period}_{i}.png", rgb)

#     return images
