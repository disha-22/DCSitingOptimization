# DCSitingOptimization

## Config File
The following components go into the config file. Config files should be stored as `.ini` extension.

`scenario_name`: the unique name defining the features of the scenario, used for storing optimization outputs and visualizations. This should also be the file name for the config file. Can be formatted as `{region}\_{demand}\_{renewables}\_{equity_type}\_{normalization}_{train_or_valid}`.

Within `scenario_name`, the following naming conventions are used.

*region* indicates what region is covered by the optimization. California, United States, only two cities (San Francisco and Los Angeles), etc.

*demand* indicates the demand profile being used. Options are (1) **flat** for flat 2GWh demand, (2) **business** for business-peaking demand with average demand 2GWh, (3) **caiso** for demand proportional to CAISO demand with average demand 2GWh.

*renewables* indicates whether the data center developer can only draw on grid electricity, or can tap into renewables. Options are (1) **withRenewables** if renewables are allowed, (2) **gridOnly** if only grid electricity is allowed. For now, we do not experiment with this and fix it to **withRenewables**.

*equity_type* indicates the evaluation metric for water inequity. Current options are (1) **max** for maximum water scarcity footprint, (2) **mad** for mean absolute difference.

*normalization* indicates the type of normalization being used. Options are (1) **all_std** to compute normalization factors as the standard deviation across regions for all sources (grid, solar, water, data center), (2) **grid_std** to compute normalization factors as the standard deviation across regions for grid electricity, (3) **max** to compute normalization factors as the maximum across regions for all sources (grid, solar, water, data center).

*train_or_valid* indicates whether we are doing a train or validation optimization problem for examining renewables complementarity. Options are (1) **train** for a train optimization instance, (2) **valid** for a validation optimization instance. 

`huc8_df`: path to the file with a footprint DataFrame.

`solar_proportion_df`: path to the file with a solar production timeseries DataFrame.

`wind_proportion_df`: path to the file with a wind production timeseries DataFrame.

`demand_profile`: path to the file with a demand profile DataFrame.

`grid_only`: indicates whether the data center developer can only draw on grid electricity, or can tap into renewables. Options are (1) **False** if renewables are allowed, (2) **True** if only grid electricity is allowed. For now, we do not experiment with this and fix it to **False**.

`equity_type`: indicates the evaluation metric for water inequity. Current options are (1) **max** for maximum water scarcity footprint, (2) **mad** for mean absolute difference.

`weights_file`: path to the file with a DataFrame of all weight combinations to try out.

`normalization`: indicates the type of normalization being used. Options are (1) **all_std** to compute normalization factors as the standard deviation across regions for all sources (grid, solar, water, data center), (2) **grid_std** to compute normalization factors as the standard deviation across regions for grid electricity, (3) **max** to compute normalization factors as the maximum across regions for all sources (grid, solar, water, data center).

You can run the optimization just by executing `run_optimization(config_path)`.