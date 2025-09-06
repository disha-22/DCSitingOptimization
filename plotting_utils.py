import matplotlib as mpl
import matplotlib.pyplot as plt
from pyproj import Transformer

# transform centroid to lat and lon
transformer = Transformer.from_crs(5070, 4326)
transformer_inv = Transformer.from_crs(4326, 5070)

# functionalize visualization of different statistics

def visualize_stats(df, col, title, cmap, cmap_label, cmap_lims=None, colorNorm=0, ax=None, interest_point_list=[], point_color='red'):
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
        _ = df.plot(column=col, edgecolor='black', linewidth=0.5, cmap=cmap, legend=True, legend_kwds={"label": cmap_label}, ax=ax)
    else:
        # _ = df.plot(column=col, edgecolor='black', linewidth=0.5, cmap=cmap, vmin=cmap_lims[0], vmax=cmap_lims[1], norm=colors.CenteredNorm(vcenter=colorNorm), legend=True, legend_kwds={"label": cmap_label}, ax=ax)
        _ = df.plot(column=col, edgecolor='black', linewidth=0.5, cmap=cmap, vmin=cmap_lims[0], vmax=cmap_lims[1], legend=True, legend_kwds={"label": cmap_label}, ax=ax)

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