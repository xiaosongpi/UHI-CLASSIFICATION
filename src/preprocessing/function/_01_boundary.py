import geopandas as gpd

def get_boundary_file(file_path, subdistrict = "Seluruh kecamatan"):
    boundary = gpd.read_file(file_path)
    boundary = boundary[boundary["WADMPR"] == "Daerah Khusus Ibukota Jakarta"]

    if subdistrict == "Seluruh kecamatan":
        boundary = boundary[boundary['WADMKK'] != 'Administrasi Kepulauan Seribu']
    else:
        boundary = boundary[boundary['WADMKC'] == subdistrict]

    return boundary.reset_index(drop = True)

boundary = get_boundary_file(
    "data/raw_data/boundary.gpkg"
)

boundary.to_file(
    "data/raw_data/boundary_clean.gpkg",
    driver="GPKG"
)