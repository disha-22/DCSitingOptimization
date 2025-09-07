import pandas as pd
import numpy as np
import xarray as xr
import h5pyd
from scipy.spatial import cKDTree


def nearest_site_solar(tree, lat_coord, lon_coord):
    """ 
    Find the nearest site in the solar dataset.

    Parameters
    ----------
        tree: scipy.spatial.cKDTree
            Lookup tree for nearest neighbor
        lat_coord: float
            Latitude
        lon_coord: float
            Longitude

    Returns
    -------
        Index of closest point in the solar dataset
    """

    lat_lon = np.array([lat_coord, lon_coord])
    _, pos = tree.query(lat_lon)
    return pos


def download_solar(year, geo_df):
    """ 
    Download solar direct normal irradiance (DNI), diffuse horizontal irradiance, global horizontal irradiance for the specified year.

    Parameters
    ----------
        year: string
            Year to download data for.
        geo_df: gpd.GeoDataFrame
            GeoDataFrame containing locations of interest where DNI, DHI, and GHI data is collected from.

    Returns
    -------
        irradiation_xr: xarray.Dataset
            xr dataset with DNI, DHI, and GHI data with UTC time index.
    """

    # takes around a minute to run 

    solar = h5pyd.File(f"/nrel/nsrdb/v3/nsrdb_{year}.h5")
    meta = pd.DataFrame(solar['meta'][...])

    # find the nearest timeseries
    # code source: https://github.com/NREL/hsds-examples/blob/master/notebooks/03_NSRDB_introduction.ipynb
    dset_coords = solar['coordinates'][...]
    tree = cKDTree(dset_coords)

    # collect DNI, DHI, GHI at every point
    # takes around 4 min to run

    dni_data = solar['dni']
    dhi_data = solar['dhi']
    ghi_data = solar['ghi']

    dni_subbasins = np.zeros((dni_data.shape[0], geo_df.shape[0])) # rows: timestamps, # columns: locations
    dhi_subbasins = np.zeros((dhi_data.shape[0], geo_df.shape[0])) # rows: timestamps, # columns: locations
    ghi_subbasins = np.zeros((ghi_data.shape[0], geo_df.shape[0])) # rows: timestamps, # columns: locations

    time_offset = np.zeros(geo_df.shape[0]) # collect timezone info relative to UTC (example: California is -8 hours compared to UTC)

    # for each California subbasin centroid, get the solar data of the closest point
    for idx, (_, row) in enumerate(geo_df.iterrows()):
        centroid_idx = nearest_site_solar(tree, row['centroid_lat'], row['centroid_lon'])
        
        time_offset[idx] = meta.loc[centroid_idx, 'timezone']

        dni_tseries = dni_data[:, centroid_idx] / dni_data.attrs['psm_scale_factor']
        dhi_tseries = dhi_data[:, centroid_idx] / dhi_data.attrs['psm_scale_factor']
        ghi_tseries = ghi_data[:, centroid_idx] / ghi_data.attrs['psm_scale_factor'] # units: W/m^2. https://nsrdb.nrel.gov/data-sets/us-data

        dni_subbasins[:, idx] = dni_tseries
        dhi_subbasins[:, idx] = dhi_tseries
        ghi_subbasins[:, idx] = ghi_tseries


    # by visual inspection of the data, we can tell that this time index data is UTC timezone. (California is -8 hours, according to the metadata)
    # collect time index
    time_index = pd.to_datetime(solar['time_index'][...].astype(str))

    irradiation_xr = xr.Dataset()
    irradiation_xr['DNI'] = xr.DataArray(dni_subbasins, dims=("Time", "HUC8"), coords={"Time": time_index, "HUC8": geo_df.index})
    irradiation_xr['DHI'] = xr.DataArray(dhi_subbasins, dims=("Time", "HUC8"), coords={"Time": time_index, "HUC8": geo_df.index})
    irradiation_xr['GHI'] = xr.DataArray(ghi_subbasins, dims=("Time", "HUC8"), coords={"Time": time_index, "HUC8": geo_df.index})

    irradiation_xr.attrs['Timezone'] = time_offset

    return irradiation_xr