import ee
from shapely import force_2d
from shapely.geometry import mapping
import json
import streamlit as st

from src.preprocessing.function._01_boundary import get_boundary_file

PROJECT = "earth-engine-project-503206"

def gee_init():
    try:
        info = dict(st.secrets["gee"])
    except Exception:
        info = None

    if info:  # di Streamlit Cloud
        creds = ee.ServiceAccountCredentials(
            info["client_email"], key_data=json.dumps(info)
        )
        ee.Initialize(creds, project=info.get("project_id", PROJECT))
    else:  # di laptop
        try:
            ee.Initialize(project=PROJECT)
        except ee.EEException:
            ee.Authenticate()
            ee.Initialize(project=PROJECT)

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