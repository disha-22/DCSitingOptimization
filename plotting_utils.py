# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# geography
from pyproj import Transformer

# data
import pickle as pkl
import numpy as np
import pandas as pd
import geopandas as gpd

# helpful functions
from Model_PSCC_Draft_RC import *


# transform centroid to lat and lon
transformer = Transformer.from_crs(5070, 4326)
transformer_inv = Transformer.from_crs(4326, 5070)

# helpful data
huc8_df = gpd.read_file("Data/Footprint/california.geojson")



color_dict = {
    'Grid': 'silver',
    'Solar': 'yellow',
    'Wind': 'deepskyblue',
    'Data Center': 'red'
}


# =========================== pie chart dictionaries ============================
# pie chart: size is the total electricity usage. sectors are how much electricity usage comes from grid, solar, wind.
usage_dict = {'Grid': 'Total_Grid_MWh',
              'Solar': 'Solar_Used_MWh', 
              'Wind': 'Wind_Used_MWh'}

# pie chart: size is the total electricity curtailment. sectors are how much curtailed electricity comes from solar and wind.
curtail_dict = {'Solar': 'Solar_Curtailed_MWh',
                'Wind': 'Wind_Curtailed_MWh'}

# pie chart: size is the total water scarcity footprint. sectors are how much water scarcity footprint comes from grid, solar, wind, data center.
water_dict = {'Grid': 'Total Grid Water Scarcity Footprint [m^3-eq]',
              'Solar': 'Total Solar Water Scarcity Footprint [m^3-eq]',
              'Wind': 'Total Wind Water Scarcity Footprint [m^3-eq]',
              'Data Center': 'Total Data Center Water Scarcity Footprint [m^3-eq]'}

# pie chart: size is the total emissions footprint. sectors are how much emissions footprint comes from grid, solar, and wind
emissions_dict = {'Grid': 'Total Grid Emissions [tons CO2-eq]',
                  'Solar': 'Total Solar Emissions [tons CO2-eq]',
                  'Wind': 'Total Wind Emissions [tons CO2-eq]'}

# pie chart: size is the total cost. sectors are how much cost comes from grid, solar, wind, and data center
cost_dict = {'Grid': 'Total Grid Cost [$]',
             'Solar': 'Total Solar Cost [$]',
             'Wind': 'Total Wind Cost [$]',
             'Data Center': 'Total Data Center Cost [$]'}


# if pie charts get too complicated, place it into a class structure.
# other resources: https://www.geeksforgeeks.org/python/how-to-set-border-for-wedges-in-matplotlib-pie-chart/
def geoplot_pie(df, lat_col, lon_col, category_dict, unit_factor, unit, pie_scale, ax=None, num_circles=3, size_lon=-125.5, size_lat=31.6, vspace=1.7):
    """ 
    Plot a map, with pie charts at the centroid of each region. Area of pie chart is proportional to the total amount.

    Parameters
    ----------
        df: gpd.GeoDataFrame
            GeoDataFrame with data of interest.
        lat_col: string
            Name of column with latitude coordinates.
        lon_col: string
            Name of column with longitude coordinates.
        category_dict: dictionary
            Dictionary with sources (grid, solar, wind, data center) as keys, and column names as values.
        unit_factor: float
            Conversion factor from measurements to unit of measurement.
        unit: string
            Unit of measurement.
        pie_scale: float
            Scaling factor for pie size.
        ax: matplotlib.axes.Axes
            Axes to plot the map and pie chart on.
        num_circles: int
            Number of size circles for the legend.
        size_lon: float
            Longitude for size legend.
        size_lat: float
            Latitude of the lowest size legend.
        vspace: float
            Vertical spacing between size legends.
            
    Returns
    -------
        None
    """

    if ax is None:
        ax = plt.gca() # get current axis

    df.plot(ax=ax, edgecolor='black', facecolor='white', linewidth=0.5) # background map

    L, R = ax.get_xlim()
    B, T = ax.get_ylim()

    # plot the pie charts
    for _, row in df.iterrows():

        if row[category_dict.values()].sum() > 0:
            # coordinates, in units of degrees
            lon = row[lon_col]
            lat = row[lat_col]

            ax.pie(row[category_dict.values()], radius=np.sqrt(row[category_dict.values()].sum()) * pie_scale, \
                center=(lon, lat), colors=[color_dict[key] for key in category_dict.keys()], wedgeprops={
                    'edgecolor': 'black',
                    'linewidth': 1
                })

    # make the pie size legend
    ref_sizes = np.linspace(0, df[category_dict.values()].sum(axis=1).max(), num_circles+1)[1:] # get num_circles reference sizes
    biggest = np.sqrt(ref_sizes[-1]) * pie_scale

    for size, (lon, lat) in zip(ref_sizes, [(size_lon, size_lat + 2 * vspace), (size_lon, size_lat + vspace), (size_lon, size_lat)][3-num_circles:]):
    #     ref_label = f'{size * unit_factor:.1f}' + unit
        radius = np.sqrt(size) * pie_scale

    #     # code source: https://stackoverflow.com/questions/33094509/correct-sizing-of-markers-in-scatter-plot-to-a-radius-r-in-matplotlib

        # multiplier = (R - L)/(7 * 0.8)
        # points_whole_ax = 7 * 0.8 * 72
        # # points_whole_ax = (R - L) * 0.8 * 72    # 1 point = dpi / 72 pixels
        # points_radius = 2 * radius / 1.0 * points_whole_ax
        # ax.scatter(lon, lat, s=points_radius**2, color='r', alpha=0.5)
        # ax.scatter(lon, lat, s=(radius*100)**2, color='r', alpha=0.5)


        e = Ellipse(xy=(lon, lat), width=radius*2, height=radius*2, angle=0, alpha=0.5, color='gray')

        ax.add_artist(e)

        ax.text(lon+biggest+0.2, lat-0.2, f"{size * unit_factor:.0f}" + unit)
    
    # make the color legend
    for key in category_dict.keys():
        ax.bar(0, 0, color=color_dict[key], label=key, zorder=-3)

    # ax.set_xlim(L - 3, R)
    # ax.set_ylim(B - 3, T + 0.5)
    ax.set_xlim(lon - 1, R)
    ax.set_ylim(lat - 1, T + 0.5)

    ax.legend(loc='upper right', fontsize=15)
    

# functionalize visualization of different statistics

def visualize_stats(df, col, title, cmap, cmap_label, cmap_lims=None, colorNorm=0, shrink=0.75, ax=None, interest_point_list=[], point_color='red'):
    """ 
    Visualization of selected statistics on HUC8 subbasins.

    Parameters
    ----------
        df: geopandas.GeoDataFrame
            GeoDataFrame to visualize
        col: column 
            Column of GeoDataFrame for colors
        title: string
            Title of map
        cmap: string
            Color map to use
        cmap_label: string
            Label of color bar
        cmap_lims: list
            List of two values for color limits
        colorNorm: float
            Value around which to center the coloring
        shrink: float
            Factor by which to shrink the color bar
        ax: matplotlib.axes.Axes
            Axis to plot the map on.
        interest_point_list: list
            List of point coordinates to plot
        point_color: string
            Color of points

    Returns
    -------
        ax: matplotlib.axes.Axes
            Axis with the plotted figure
    """

    mpl.rcParams.update({'font.size': 20})

    if ax is None:
        _, ax = plt.subplots()

    if cmap_lims is None:
        _ = df.plot(column=col, edgecolor='black', linewidth=0.5, cmap=cmap, legend=True, legend_kwds={"label": cmap_label, "shrink": shrink}, ax=ax)
    else:
        # _ = df.plot(column=col, edgecolor='black', linewidth=0.5, cmap=cmap, vmin=cmap_lims[0], vmax=cmap_lims[1], norm=colors.CenteredNorm(vcenter=colorNorm), legend=True, legend_kwds={"label": cmap_label}, ax=ax)
        _ = df.plot(column=col, edgecolor='black', linewidth=0.5, cmap=cmap, vmin=cmap_lims[0], vmax=cmap_lims[1], legend=True, legend_kwds={"label": cmap_label, "shrink": shrink}, ax=ax)

    ax.set_title(title)
    ax.set_axis_off()

    # plot points of interest
    for point in interest_point_list:
        point_translated = transformer_inv.transform(point[0], point[1]) # convert lat lon to CRS of interest
        ax.scatter(point_translated[0], point_translated[1], marker=(5,1), s=100, color=point_color)

    # code source for scale bar: https://geopandas.org/en/stable/gallery/matplotlib_scalebar.html
    # points = gpd.GeoSeries(
    #     [Point(-73.5, 40.5), Point(-74.5, 40.5)], crs=4326
    # )  # Geographic WGS 84 - degrees
    # points = points.to_crs(5070)
    # distance_meters = points[0].distance(points[1])
    # ax.add_artist(ScaleBar(100, "m", location="lower left"))

    return ax





def curtailed_renewables(config_path, results_path):
    """ 
    Compute the level of used versus curtailed solar and wind.

    Parameters
    ----------
        config_path: string
            Path to config file.
        results_path: string
            Path to results.

    Returns
    -------
        solar_used: np.ndarray
            Hourly data of how much solar energy was used to power data centers.
        solar_curtailed: np.ndarray
            Hourly data of how much solar energy was curtailed.
        wind_used: np.ndarray
            Hourly data of how much wind energy was used to power data centers.
        wind_curtailed: np.ndarray
            Hourly data of how much wind energy was curtailed.
    """

    config = import_config(config_path)

    # read data
    huc8_df = gpd.read_file(config['huc8_df'])
    solar_proportion_df = pd.read_csv(config['solar_proportion_df'], index_col=0)
    wind_proportion_df = pd.read_csv(config['wind_proportion_df'], index_col=0)
    demand_profile = pd.read_csv(config['demand_profile'], index_col=0)

    data = prepare_optimization_data(
        huc8_df, solar_proportion_df, wind_proportion_df, demand_profile
    )


    # read results
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    

    # calculate hourly renewables
    solar_hourly = np.diag(results['results']['s'].flatten()) @ data['C_s']
    wind_hourly = np.diag(results['results']['w'].flatten()) @ data['C_w']

    total_renewables = solar_hourly + wind_hourly

    renewables_divide = np.where(total_renewables > 0, total_renewables, 1) # prepare an np.ndarray for division

    # derive scaled down version of renewables production - accounts for curtailing

    solar_used = np.where(total_renewables >= results['results']['a'], (results['results']['a']/renewables_divide) * solar_hourly, solar_hourly)
    solar_curtailed = solar_hourly - solar_used
    wind_used = np.where(total_renewables >= results['results']['a'], (results['results']['a']/renewables_divide) * wind_hourly, wind_hourly)
    wind_curtailed = wind_hourly - wind_used

    return solar_used, solar_curtailed, wind_used, wind_curtailed


# enhance the results gdf
def enhance_results(config_path, results_path):
    """ 
    Add in columns with extra information for the optimized results GeoDataFrame.

    Parameters
    ----------
        config_path: string
            Path to config file.
        results_path: string
            Path to results.
    
    Returns
    -------
        results_gdf: gpd.GeoDataFrame
            GeoDataFrame with summary of optimized results.
    """

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    solar_used, solar_curtailed, wind_used, wind_curtailed = curtailed_renewables(config_path, results_path)

    results_df = results['results_df']

    # add in the geometry and footprints
    results_gdf = huc8_df.merge(results_df, left_on="HUC8", right_on="HUC8")

    # renewables: used energy and curtailed energy
    results_gdf['Solar_Used_MWh'] = np.sum(solar_used, axis=1)
    results_gdf['Solar_Curtailed_MWh'] = np.sum(solar_curtailed, axis=1)
    results_gdf['Wind_Used_MWh'] = np.sum(wind_used, axis=1)
    results_gdf['Wind_Curtailed_MWh'] = np.sum(wind_curtailed, axis=1)

    # fix data center capacity factor
    results_gdf['Data_Center_Capacity_Factor'] = results_gdf['Total_Demand_MWh'].where(results_df['Total_Demand_MWh'] > 1, 0) / (results_df['New_Capacity_MW'] * 8760 + 1e-10)

    # water scarcity
    results_gdf['Total Grid Water Scarcity Footprint [m^3-eq]'] = results_gdf['Total_Grid_MWh'] * results_gdf['Grid Water Scarcity Footprint [m3-eq/MWh]']
    results_gdf['Total Solar Water Scarcity Footprint [m^3-eq]'] = results_gdf['Solar_MWh'] * results_gdf['Solar Water Scarcity Footprint [m3-eq/MWh]']
    results_gdf['Total Wind Water Scarcity Footprint [m^3-eq]'] = results_gdf['Wind_MWh'] * results_gdf['Wind Water Scarcity Footprint [m3-eq/MWh]']
    results_gdf['Total Data Center Water Scarcity Footprint [m^3-eq]'] = results_gdf['Total_Demand_MWh'] * results_gdf['Data Center Water Scarcity Footprint [m3-eq/MWh]']
    results_gdf['Total Water Scarcity Footprint [m3-eq]'] = results_gdf['Total Grid Water Scarcity Footprint [m^3-eq]'] +  results_gdf['Total Solar Water Scarcity Footprint [m^3-eq]'] + results_gdf['Total Wind Water Scarcity Footprint [m^3-eq]'] + results_gdf['Total Data Center Water Scarcity Footprint [m^3-eq]']

    # emissions
    results_gdf['Total Grid Emissions [tons CO2-eq]'] = results_gdf['Total_Grid_MWh'] * results_gdf['Grid Carbon Footprint [tons CO2-eq/MWh]']
    results_gdf['Total Solar Emissions [tons CO2-eq]'] = results_gdf['Solar_MWh'] * results_gdf['Solar Carbon Footprint [tons CO2-eq/MWh]']
    results_gdf['Total Wind Emissions [tons CO2-eq]'] = results_gdf['Wind_MWh'] * results_gdf['Wind Carbon Footprint [tons CO2-eq/MWh]']
    results_gdf['Total Emissions [tons CO2-eq]'] = results_gdf['Total Grid Emissions [tons CO2-eq]'] + results_gdf['Total Solar Emissions [tons CO2-eq]'] + results_gdf['Total Wind Emissions [tons CO2-eq]']

    # cost
    results_gdf['Total Grid Cost [$]'] = results_gdf['Total_Grid_MWh'] * results_gdf['Electricity Price [$/MWh]']
    results_gdf['Total Solar Cost [$]'] = results_gdf['Solar_MWh'] * results_gdf['Mean Solar LCOE [$/MWh]']
    results_gdf['Total Wind Cost [$]'] = results_gdf['Wind_MWh'] * results_gdf['Mean Wind LCOE [$/MWh]']
    results_gdf['Total Data Center Cost [$]'] = results_gdf['New_Capacity_MW'] * results_gdf['Data_Center_Cost_Per_MWh']
    results_gdf['Total Cost [$]'] = results_gdf['Total Grid Cost [$]'] + results_gdf['Total Solar Cost [$]'] + results_gdf['Total Wind Cost [$]'] + results_gdf['Total Data Center Cost [$]']

    # avoid large coordinates, which may cause issues with plot size
    results_gdf.to_crs('EPSG:4326', inplace=True)

    return results_gdf


def result_maps(config_path, results_path, axes=None):
    """ 
    Show maps of the results:
        - [0][0] Added data centers, colored map
        - [0][1] Capacity factors of data centers, colored map
        - [1][0] Pie chart of electricity usage on a map
        - [1][1] Pie chart of curtailed electricity on a map
        - [1][2] Pie chart of water scarcity footprint on a map

    Parameters
    ----------
        config_path: string
            Path to config file.
        results_path: string
            Path to results.
        axes: mpl.axes.Axes
            Axes to plot results on.
    """

    if axes is None:
        _, axes = plt.subplots(2, 3, figsize=(21, 14))

    # plotting data
    results_gdf = enhance_results(config_path, results_path)

    # added data centers
    visualize_stats(results_gdf, 'New_Capacity_MW', 'Added Data Center Capacity', cmap='Greens', cmap_label='MW', ax=axes[0][0])

    # capacity factor of data centers
    visualize_stats(results_gdf, 'Data_Center_Capacity_Factor', 'Data Center Capacity Factor', cmap='Blues', cmap_label='', cmap_lims=[0, 1], ax=axes[0][1]) # unitless. We can also change to percentage if we want.

    # leave [0][2] blank
    axes[0][2].set_axis_off()

    # pie chart of electricity usage
    geoplot_pie(results_gdf, "centroid_lat", "centroid_lon", usage_dict, 1e-6, " TWh", 2e-4, ax=axes[1][0])
    axes[1][0].set_title("Total Data Center Electricity Usage")

    # pie chart of curtailed electricity
    geoplot_pie(results_gdf, "centroid_lat", "centroid_lon", curtail_dict, 1e-6, " TWh", 2e-4, ax=axes[1][1])
    axes[1][1].set_title("Total Electricity Curtailment")

    # pie chart of water scarcity footprint
    geoplot_pie(results_gdf, "centroid_lat", "centroid_lon", water_dict, 1e-6, r"$\times 10^6$ m$^3$-eq", 1e-4, ax=axes[1][2])
    axes[1][2].set_title("Total Water Scarcity Footprint")


# TODO Richard bar chart summaries