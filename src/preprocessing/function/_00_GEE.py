import ee
from shapely import force_2d
from shapely.geometry import mapping

from src.preprocessing.function._01_boundary import get_boundary_file

def gee_init():
    try:
        ee.Initialize(project="earth-engine-project-503206")
    except ee.EEException:
        ee.Authenticate()
        ee.Initialize(project="earth-engine-project-503206")

def get_region(area):
    boundary = get_boundary_file("data/raw_data/boundary.gpkg", area)
    geometry = force_2d(boundary.to_crs(epsg=4326).geometry.union_all())
    return ee.Geometry(mapping(geometry))

def get_landsat_image(area, start_date, end_date):
    return (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(get_region(area))
        .filterDate(str(start_date), str(end_date))
        .filter(ee.Filter.lt("CLOUD_COVER", 30))
    )