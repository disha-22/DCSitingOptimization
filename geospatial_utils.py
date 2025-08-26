# working with data
import numpy as np



class resolve_regions:
    """ 
    Converts one subdivision of space (e.g., county) into another subdivision of space (e.g., HUC8.)

    Averages out value of interest according to amount of overlap with the first subdivision of space, or the 
    closest region. 
    """

    def __init__(self, from_df, to_df, value_col, id_col):
        """ 
        Init method for the class resolve_regions.

        Parameters
        ----------
            from_df: gpd.GeoDataFrame
                Dataframe to convert from
            to_df: gpd.GeoDataFrame
                Dataframe to convert to
            value_col: string
                Name of column with values of interest
            id_col: string
                Name of column with unique identifiers, in to_df
        """

        self.from_df = from_df
        self.to_df = to_df
        self.value_col = value_col
        self.id_col = id_col

# TODO: make test cases for how this function works?
    def weighted_computation(self, id):
        """ 
        Perform the weighted average computation for the output row with identification id.

        Parameters
        ----------
            id: string
                Unique ID for the output row
        """

        to_geometry = self.to_df[self.to_df[self.id_col] == id]['geometry'].iloc[0] # extract geometry of to_df
        from_to_intersect = self.from_df['geometry'].intersection(to_geometry) # obtain intersections

        area_sum = np.sum(from_to_intersect.area)

        if area_sum > 0: # some intersection
            weighted_val = 0 # keep track of weighted sum for value of interest

            for idx, area in from_to_intersect.area.items():
                weighted_val += self.from_df.loc[idx, self.value_col]*area
            
            weighted_val /= area_sum
        
        else:
            from_to_distance = self.from_df['geometry'].distance(to_geometry) # obtain distances
            from_to_distance.sort_values(ascending=True, inplace=True)
            lowest_idx = from_to_distance.index[0]

            weighted_val = self.from_df.loc[lowest_idx, self.value_col]

        return weighted_val

    def convert_regions(self):
        """ 
        Perform conversion between the subdivisions of space, and average the value of interest.
        """

        converted_value_list = []

        for id in self.to_df[self.id_col]: # iterate over geometries of to_df
            converted_value_list.append(self.weighted_computation(id))

        self.to_df[self.value_col] = converted_value_list # set the output values after conversion