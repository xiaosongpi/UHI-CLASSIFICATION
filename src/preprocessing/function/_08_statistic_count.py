import pandas as pd
import numpy as np
from rasterio.features import geometry_mask

def count_statistic (images, boundary):
    result = []

    for period, group_image in images.items():
        for items in group_image:
            image = items['image']
            mask_image = items['mask_image']
            transform = items['transform']
            boundary_crs = boundary.to_crs(items['crs'])

            st_b10 = image[5]
            ndvi = image[6]
            ndbi = image[7]
            mndwi = image[8]

            for subdistrict in boundary_crs.itertuples():
                subdistrict_name = subdistrict.WADMKC
                subdistrict_geometry = subdistrict.geometry

                mask = geometry_mask(
                    [subdistrict_geometry],
                    transform = transform,
                    out_shape = image.shape[1:],
                    invert = True,
                )

                valid_px = mask_image & mask & (ndvi != -9999)
                ndvi_v = ndvi[valid_px]

                valid_px = mask_image & mask & (ndbi != -9999)
                ndbi_v = ndbi[valid_px]
                
                valid_px = mask_image & mask & (mndwi != -9999)
                mndwi_v = mndwi[valid_px]

                valid_px = mask_image & mask & (st_b10 != -9999)
                st_b10_v = st_b10[valid_px]

                if (ndvi_v.size == 0 or ndbi_v.size == 0 or mndwi_v.size == 0 or st_b10_v.size == 0):
                    continue

                mean_ndvi = np.mean(ndvi_v)
                std_ndvi = np.std(ndvi_v)
                min_ndvi = np.min(ndvi_v)
                max_ndvi = np.max(ndvi_v)
                q1_ndvi = np.percentile(ndvi_v, 25)
                median_ndvi = np.median(ndvi_v)
                q3_ndvi = np.percentile(ndvi_v, 75)
                mean_ndbi = np.mean(ndbi_v)
                std_ndbi = np.std(ndbi_v)
                min_ndbi = np.min(ndbi_v)
                max_ndbi = np.max(ndbi_v)
                q1_ndbi = np.percentile(ndbi_v, 25)
                median_ndbi = np.median(ndbi_v)
                q3_ndbi = np.percentile(ndbi_v, 75)
                mean_mndwi = np.mean(mndwi_v)
                std_mndwi = np.std(mndwi_v)
                min_mndwi = np.min(mndwi_v)
                max_mndwi = np.max(mndwi_v)
                q1_mndwi = np.percentile(mndwi_v, 25)
                median_mndwi = np.median(mndwi_v)
                q3_mndwi = np.percentile(mndwi_v, 75)
                
                valid_px = mask_image & mask & (ndvi != -9999) & (ndbi != -9999) & (mndwi != -9999) & (st_b10 != -9999)

                total_px = np.count_nonzero(mask)
                valid_count = np.count_nonzero(valid_px)
                valid_percent = (valid_count / total_px * 100) if total_px > 0 else 0

                if valid_percent < 10:
                    print(f"DIKECUALIKAN: {subdistrict_name} | {period}")
                    continue

                result.append({
                    'period': period,
                    'subdistrict': subdistrict_name,
                    'ndvi_mean': mean_ndvi,
                    'ndvi_std': std_ndvi,
                    'ndvi_min': min_ndvi,
                    'ndvi_max': max_ndvi,
                    'ndvi_q1': q1_ndvi,
                    'ndvi_median': median_ndvi,
                    'ndvi_q3': q3_ndvi,
                    'ndbi_mean': mean_ndbi,
                    'ndbi_std': std_ndbi,
                    'ndbi_min': min_ndbi,
                    'ndbi_max': max_ndbi,
                    'ndbi_q1': q1_ndbi,
                    'ndbi_median': median_ndbi,
                    'ndbi_q3': q3_ndbi,
                    'mndwi_mean': mean_mndwi,
                    'mndwi_std': std_mndwi,
                    'mndwi_min': min_mndwi,
                    'mndwi_max': max_mndwi,
                    'mndwi_q1': q1_mndwi,
                    'mndwi_median': median_mndwi,
                    'mndwi_q3': q3_mndwi,
                    'lst': st_b10_v,
                })

                print(
                    f"{period} | {subdistrict_name} | "
                    f"Valid: {valid_count:,}/{total_px:,} "
                    f"({valid_percent:.2f}%)"
                )

    df = pd.DataFrame(result)

    return df