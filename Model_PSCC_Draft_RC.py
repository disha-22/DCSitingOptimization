# optimization tools
import cvxpy as cp

# working with data
import numpy as np
import pandas as pd
import geopandas as gpd
import pickle
import configparser

# tracking
import wandb

# files
import os

def prepare_optimization_data(huc8_df, solar_proportion_df, wind_proportion_df,
                            demand_profile, data_center_cost=6e5, train_path=None):
    """
    Prepare all data matrices for the optimization problem

    Parameters
    ----------
        huc8_df: gpd.GeoDataFrame
            GeoDataFrame with location-specific prices, water scarcity footprint, and emissions footprint data
        solar_proportion_df: pd.DataFrame
            DataFrame with time series of solar production
        wind_proportion_df: pd.DataFrame
            DataFrame with time series of wind production
        demand_profile: np.ndarray
            Numpy array with time series of total data center demand profile
        data_center_cost: float
            Cost of data center, in ($/MWh-year)
        train_path: string
            Whether to include data from a trained optimization file. If it's a string, then it should point to a path of `train' results.

    Returns
    -------
        Dictionary with input optimization data variables:
            L: number of locations
            T: number of time steps
            P_dc: price of data center in each location
            P_g: price of grid electricity in each location
            P_s: price of solar electricity in each location
            P_w: price of wind electricity in each location
            S_dc: water scarcity footprint of data center in each location
            S_g: water scarcity footprint of grid in each location
            S_s: water scarcity footprint of solar in each location
            S_w: water scarcity footprint of wind in each location
            E_g: emissions footprint of grid in each location
            E_s: emissions footprint of solar in each location
            E_w: emissions footprint of wind in each location
            C_s: proportion of solar output at time t for each location
            C_w: proportion of wind output at time t for each location
            D: demand profile
            Y: existing data center capacity in each location
            huc8_order: order of HUC8 for each data array
            (if train_path was inputted: we can return the optimized results for renewables validation.)
            x: optimized data center capacity
            s: optimized solar capacity
            w: optimized wind capacity
    """

    L = len(huc8_df)
    T = len(demand_profile)

    # fix an order in which to count the HUC8 regions
    huc8_order = huc8_df['HUC8'].values

    # Prepare cost data
    P_dc = np.ones(L) * data_center_cost  # Cost of new data center capacity [$/MW]
    P_g = huc8_df['Electricity Price [$/MWh]'].values  # Grid electricity cost [$/MWh]
    P_s = huc8_df['Mean Solar LCOE [$/MWh]'].values  # Solar LCOE [$/MWh]
    P_w = huc8_df['Mean Wind LCOE [$/MWh]'].values  # Wind LCOE [$/MWh]

    # match order between price and footprint DF
    footprint_ordered = huc8_df.set_index('HUC8_str').reindex(huc8_order).reset_index()

    # Prepare water scarcity footprint data
    S_dc = footprint_ordered['Data Center Water Scarcity Footprint [m3-eq/MWh]'].values  # DC water scarcity [m³-eq/MWh]
    S_g = footprint_ordered['Grid Water Scarcity Footprint [m3-eq/MWh]'].values  # Grid water scarcity [m³-eq/MWh]
    S_s = footprint_ordered['Solar Water Scarcity Footprint [m3-eq/MWh]'].values  # Solar water scarcity [m³-eq/MWh]
    S_w = footprint_ordered['Wind Water Scarcity Footprint [m3-eq/MWh]'].values  # Wind water scarcity [m³-eq/MWh]

    # Prepare carbon footprint data
    E_g = footprint_ordered['Grid Carbon Footprint [tons CO2-eq/MWh]'].values  # Grid emissions [tons CO2-eq/MWh]
    E_s = footprint_ordered['Solar Carbon Footprint [tons CO2-eq/MWh]'].values  # Solar emissions [tons CO2-eq/MWh]
    E_w = footprint_ordered['Wind Carbon Footprint [tons CO2-eq/MWh]'].values  # Wind emissions [tons CO2-eq/MWh]

    # read solar and wind proportion data (and reshape to (HUC8 subbasin, hour))
    C_s = solar_proportion_df[huc8_order].values.T
    C_w = wind_proportion_df[huc8_order].values.T

    # normalize such that each solar and wind row sums to 1 and indicates a proportion
    C_s = C_s / (C_s.sum(axis=1, keepdims=True) + 1e-10)
    C_w = C_w / (C_w.sum(axis=1, keepdims=True) + 1e-10)

    # Demand profile
    D = demand_profile.values.flatten()

    # Existing capacity (assume zero for now, can be updated)
    Y = np.zeros((L, T)) # we reshape to (L, T) in order to ensure easier broadcasting within the optimization model.

    # Validation results
    if train_path is not None:
        # read in values for x, s, w
        with open(train_path, "rb") as f:
            train_results = pickle.load(f)

        x = train_results['results']['x']
        s = train_results['results']['s']
        w = train_results['results']['w']

        return {
            'L': L, 'T': T,
            'P_dc': P_dc, 'P_g': P_g, 'P_s': P_s, 'P_w': P_w,
            'S_dc': S_dc, 'S_g': S_g, 'S_s': S_s, 'S_w': S_w,
            'E_g': E_g, 'E_s': E_s, 'E_w': E_w,
            'C_s': C_s, 'C_w': C_w,
            'D': D, 'Y': Y,
            'huc8_order': huc8_order,
            'x': x,
            's': s,
            'w': w
        }
    else:
        return {
            'L': L, 'T': T,
            'P_dc': P_dc, 'P_g': P_g, 'P_s': P_s, 'P_w': P_w,
            'S_dc': S_dc, 'S_g': S_g, 'S_s': S_s, 'S_w': S_w,
            'E_g': E_g, 'E_s': E_s, 'E_w': E_w,
            'C_s': C_s, 'C_w': C_w,
            'D': D, 'Y': Y,
            'huc8_order': huc8_order
        }


def compute_composite_costs(data, alpha, beta, gamma, normalization='all_std'):
    """
    Compute composite cost matrices M_g, M_s, M_w according to equations (1-3).

    Parameters
    ----------
        alpha: float
            Weighting factor for water scarcity footprint.
        beta: float
            Weighting factor for monetary cost.
        gamma: float
            Weighting factor for emissions footprint.
        normalization: string
            Which type of normalization to use. 
            Options: 
                - 'grid_std' is calculating standard deviation by grid. 
                - 'all_std' is calculating standard deviation with all data.
                - 'all_max' is calculating maximum with all data.

    Returns
    -------
        M_g: pd.Series
            Composite cost for grid
        M_s: pd.Series
            Composite cost for solar
        M_w: pd.Series
            Composite cost for wind
        norm_S: float
            Normalization factor for water scarcity footprint
        norm_P: float
            Normalization factor for monetary cost
        norm_E: float
            Normalization factor for emissions footprint
    """

    all_S = np.concatenate([data['S_g'], data['S_s'], data['S_w'], data['S_dc']], axis=0)
    all_P = np.concatenate([data['P_g'], data['P_g'], data['P_w'], data['P_dc']/8760], axis=0) # because data['P_dc'] is in units of $/(MW-year), multiply by (1 year/8760 h) to standardize to units of $/MWh.
    all_E = np.concatenate([data['E_g'], data['E_s'], data['E_w'], np.zeros((data['E_g'].shape[0],))], axis=0) # note that data center emissions are 0 for any location.

    # different types of normalization
    if normalization == 'grid_std':
        norm_S = np.std(data['S_g'])
        norm_P = np.std(data['P_g'])
        norm_E = np.std(data['E_g'])
    elif normalization == 'all_std':
        norm_S = np.std(all_S)
        norm_P = np.std(all_P)
        norm_E = np.std(all_E)
    elif normalization == 'all_max':
        norm_S = np.max(all_S)
        norm_P = np.max(all_P)
        norm_E = np.max(all_E)
    else:
        raise Exception(f"Unrecognized normalization {normalization}.")

    # Avoid division by zero
    norm_S = max(norm_S, 1e-10)
    norm_P = max(norm_P, 1e-10)
    norm_E = max(norm_E, 1e-10)

    # Compute composite costs
    M_g = alpha * (data['S_g'] / norm_S) + beta * (data['P_g'] / norm_P) + gamma * (data['E_g'] / norm_E)
    M_s = alpha * (data['S_s'] / norm_S) + beta * (data['P_s'] / norm_P) + gamma * (data['E_s'] / norm_E)
    M_w = alpha * (data['S_w'] / norm_S) + beta * (data['P_w'] / norm_P) + gamma * (data['E_w'] / norm_E)

    return M_g, M_s, M_w, norm_S, norm_P, norm_E


def optimize_data_center_siting(data, scenario_name, weights_dict, equity_type='max', verbose=True, grid_only=False, normalization='all_std', train_path=None):
    """
    Solve the data center siting optimization problem

    Parameters:
    -----------
    data : dict
        Dictionary containing all prepared data
    scenario_name: string
        Name of scenario being computed
    weights_dict: dict
        Dictionary of weights, where keys are weights names and values are dictionaries of (alpha, beta, gamma, delta) weights
    equity_type : str
        'max' for maximum water scarcity or 'mad' for mean absolute difference
    verbose: boolean
        True if we want to print updates as optimization is in progress.
    grid_only: boolean
        If True, only allow use of grid electricity
    normalization: string
        Which type of normalization to use. 
        Options: 
            - 'grid_std' is calculating standard deviation by grid. 
            - 'all_std' is calculating standard deviation with all data.
            - 'all_max' is calculating maximum with all data.
    train_path: string
        Path to results from a trained optimization model. Applies when we are running a validation case.    
    """

    # ==================== Parameters =====================
    # set alpha, beta, gamma, delta as parameters (speeds up solving if we do multiple runs with different parameters: https://www.cvxpy.org/tutorial/intro/index.html#parameters)
    alpha = cp.Parameter(nonneg=True)
    beta = cp.Parameter(nonneg=True)
    gamma = cp.Parameter(nonneg=True)
    delta = cp.Parameter(nonneg=True)


    # ==================== Data =======================
    # L (number of locations) and T (number of time steps) constants
    L, T = data['L'], data['T']

    # Compute composite costs
    M_g, M_s, M_w, norm_S, norm_P, norm_E = compute_composite_costs(data, alpha, beta, gamma, normalization=normalization)


    # ==================== Decision variables =====================
    # Keep the vectors as (L, 1) shape rather than (L,) for easy broadcasting.
    x = cp.Variable((L, 1), nonneg=True)  # New DC capacity [MW]
    a = cp.Variable((L, T), nonneg=True)  # DC demand allocation [MWh]
    g = cp.Variable((L, T), nonneg=True)  # Grid power [MWh]
    s = cp.Variable((L, 1), nonneg=True)  # Annual solar [MWh]
    w = cp.Variable((L, 1), nonneg=True)  # Annual wind [MWh]


    # ======================== Water Inequity ========================
    # Compute water scarcity vector S (equation 4)
    S = (cp.diag(data['S_g']) @ cp.sum(g, axis=1, keepdims=True) +
         cp.diag(data['S_s']) @ s +
         cp.diag(data['S_w']) @ w +
         cp.diag(data['S_dc']) @ cp.sum(a, axis=1, keepdims=True))

    # Water inequity term
    if equity_type == 'max': # max water scarcity footprint
        f_equity = cp.max(S)
        equity_constraints = []
    elif equity_type == 'mad':  # mean absolute difference of water scarcity footprint
        f_equity = cp.sum(cp.abs(S - S.T)) / (L * L)
        equity_constraints = []

        # diff = cp.Variable((L, L), nonneg=True)
        # f_equity = cp.sum(diff) / (L * L)
        # for i in range(L):
        #     for j in range(L):
        #         equity_constraints.append(diff[i, j] >= S[i] - S[j])
        #         equity_constraints.append(diff[i, j] >= S[j] - S[i])
    else:
        raise Exception(f"Equity type {equity_type} not recognized.")

    # ======================= Constraints ======================
    constraints = []

    # Enforce fixed variables for validation optimization models
    train_constraints = []
    if train_path is not None:
        train_constraints.append(x == data['x'])
        train_constraints.append(s == data['s'])
        train_constraints.append(w == data['w'])
    constraints.extend(train_constraints)

    # Meet demand (equation 7)
    constraints.append(cp.sum(a, axis=0) >= data['D'])

    # Power balance (equation 8)
    constraints.append(g + cp.diag(s) @ data['C_s'] + cp.diag(w) @ data['C_w']  >= a)

    # Capacity constraint (equation 9)

    # constraints.append((x + data['Y']).reshape((data['Y'].shape[0], 1)) @ np.ones((1,T)) >= a)
    constraints.append(x + data['Y'] >= a)

    # Add equity constraints
    constraints.extend(equity_constraints)

    # Case with only grid electricity
    if grid_only:
        for l in range(L):
            constraints.append(s[l] == 0)
            constraints.append(w[l] == 0)


    # ==================== Objective function (equation 6) ====================
    obj = ((beta / norm_P) * (data['P_dc'].T @ x) +
           M_g.T @ cp.sum(g, axis=1) +
           M_s.T @ s +
           M_w.T @ w +
           (alpha / norm_S) * (data['S_dc'].T @ cp.sum(a, axis=1)) +
           (delta / norm_S) * f_equity)

    # Create and solve problem
    problem = cp.Problem(cp.Minimize(obj), constraints)


    # try all different parameter combinations
    for _, params in weights_dict.items():
        weights_name = params['weights_name']
        alpha.value = params['alpha']
        beta.value = params['beta']
        gamma.value = params['gamma']
        delta.value = params['delta']

        # update params
        params['equity_type'] = equity_type
        params['grid_only'] = grid_only
        params['huc8_order'] = data['huc8_order']

        problem.solve(solver=cp.GUROBI if 'GUROBI' in cp.installed_solvers() else cp.ECOS, verbose=verbose)

        if problem.status not in ['optimal', 'optimal_inaccurate']:
            print(f"Warning: Problem status is {problem.status}")

        # Extract results
        results = {
            'x': x.value,  # New DC capacity
            'a': a.value,  # Demand allocation
            'g': g.value,  # Grid usage
            's': s.value,  # Solar usage
            'w': w.value,  # Wind usage
            'S': S.value if hasattr(S, 'value') else None,  # Water scarcity by location
            'Water_Inequity': f_equity.value,
            'Objective_Value': problem.value,
            'status': problem.status,
            'scenario_name': weights_name,
        }

        # add in the parameters
        results.update(params)


        # add insightful total metrics results
        results_df, total_metrics = analyze_results(results, data)

        print(f"Total New Capacity: {total_metrics['Total_New_Capacity_MW']:.1f} MW")
        print(f"Renewable Energy: {total_metrics['Renewable_Percent']:.1f}%")
        print(f"Water Inequity: {total_metrics['Water_Inequity']:.2f}")
        print(f"Total Emissions: {total_metrics['Total_Emissions_tonsCO2']:.0f} tons CO2")
        print(f"Total Water Scarcity: {total_metrics['Total_Water_Scarcity_m3eq']:.0f} m³-eq")


        # log statistics in wandb
        with wandb.init(project='DCSitingOptimization', name=f"{scenario_name}/{weights_name}", config=params) as run:
            run.log(total_metrics)


        # Store results
        # all_results[scenario['name']] = {
        #     'results': results,
        #     'results_df': results_df,
        #     'metrics': metrics
        # }


        # save all relevant results
        if not os.path.exists(f"Data/Models/{scenario_name}"):
            os.mkdir(f"Data/Models/{scenario_name}/")

        with open(f"Data/Models/{scenario_name}/{weights_name}_alpha_{params['alpha']}_beta_{params['beta']}_gamma_{params['gamma']}_delta_{params['delta']}.pkl", "wb") as f:
            pickle.dump({'results': results,
                       'results_df': results_df,
                       'metrics': total_metrics}, f)


def analyze_results(results, data):
    """
    Analyze and visualize optimization results.

    Parameters
    ----------
        results: dictionary
            Dictionary of results (x, a, g, s, w).
        data: dictionary
            Dictionry of input data for the optimization problem.

    Returns
    -------
        results_df: pd.DataFrame
            DataFrame of location-specific summaries.

    """

    # ======================================== Create results dataframe ========================================
    results_df = pd.DataFrame({
        'HUC8': data['huc8_order'],
        'New_Capacity_MW': results['x'].flatten(),
        'Total_Grid_MWh': np.sum(results['g'], axis=1),
        'Solar_MWh': results['s'].flatten(),
        'Wind_MWh': results['w'].flatten(),
        'Total_Demand_MWh': np.sum(results['a'], axis=1),
        'Data_Center_Cost_Per_MWh': data['P_dc']
    })

    # Calculate percentages
    total_energy = results_df['Total_Grid_MWh'] + results_df['Solar_MWh'] + results_df['Wind_MWh']

    # note that there may be curtailed energy from solar and wind. We can refine the metric later if needed.
    results_df['Grid_Percent'] = 100 * results_df['Total_Grid_MWh'] / (total_energy + 1e-10)
    results_df['Solar_Percent'] = 100 * results_df['Solar_MWh'] / (total_energy + 1e-10)
    results_df['Wind_Percent'] = 100 * results_df['Wind_MWh'] / (total_energy + 1e-10)
    results_df['Data_Center_Capacity_Factor'] = results_df['Total_Demand_MWh'].where(results_df['Total_Demand_MWh'] > 1, 0) / (results_df['New_Capacity_MW'] * 8760 + 1e-10)

    # ======================================= Calculate total metrics =======================================
    total_metrics = {
        'Total_New_Capacity_MW': np.sum(results['x']),
        'Total_Grid_MWh': np.sum(results['g']),
        'Total_Solar_MWh': np.sum(results['s']),
        'Total_Wind_MWh': np.sum(results['w']),
        'Water_Inequity': results['Water_Inequity'],
        'Objective_Value': results['Objective_Value']
    }

    # Calculate environmental impacts
    total_emissions = (data['E_g'].T @ np.sum(results['g'], axis=1) +
                       data['E_s'].T @ results['s'] +
                       data['E_w'].T @ results['w'])[0]
        
    total_water = (data['S_g'].T @ np.sum(results['g'], axis=1) +
                   data['S_s'].T @ results['s'] + 
                   data['S_w'].T @ results['w'])[0]
        
    total_metrics['Total_Emissions_tonsCO2'] = total_emissions
    total_metrics['Total_Water_Scarcity_m3eq'] = total_water

    # Renewable percentage
    total_energy_all = total_metrics['Total_Grid_MWh'] + total_metrics['Total_Solar_MWh'] + total_metrics['Total_Wind_MWh']
    total_metrics['Solar_Percent'] = 100 * total_metrics['Total_Solar_MWh'] / total_energy_all
    total_metrics['Wind_Percent'] = 100 * total_metrics['Total_Wind_MWh'] / total_energy_all
    total_metrics['Renewable_Percent'] = total_metrics['Solar_Percent'] + total_metrics['Wind_Percent']

    return results_df, total_metrics


def import_config(config_file):
    """ 
    Imports configuration file.

    Parameters
    ----------
        config_file: string
            Path to config file.

    Returns
    -------
        config: dictionary
            Dictionary with configuration details.
    """

    content = configparser.ConfigParser()
    content.read(config_file)
    config = {'scenario_name': content['DEFAULT']['scenario_name'], 
              'huc8_df': content['DEFAULT']['huc8_df'],
              'solar_proportion_df': content['DEFAULT']['solar_proportion_df'],
              'wind_proportion_df': content['DEFAULT']['wind_proportion_df'],
              'demand_profile': content['DEFAULT']['demand_profile'],
              'grid_only': content['DEFAULT'].getboolean('grid_only'),
              'equity_type': content['DEFAULT']['equity_type'],
              'weights_file': content['DEFAULT']['weights_file'],
              'normalization': content['DEFAULT']['normalization'],
              'train_path': content['DEFAULT'].get('train_path', None)}
    
    return config

def run_optimization(config_file):
    """
    Run the complete optimization pipeline.

    Parameters
    ----------
        config_file: string
            Path to config file
    """
    
    config = import_config(config_file)

    print(f"\n{'='*50}")
    print(f"Running: {config['scenario_name']}")
    print(f"{'='*50}")

    # read data
    huc8_df = gpd.read_file(config['huc8_df'])
    solar_proportion_df = pd.read_csv(config['solar_proportion_df'], index_col=0)
    wind_proportion_df = pd.read_csv(config['wind_proportion_df'], index_col=0)
    demand_profile = pd.read_csv(config['demand_profile'], index_col=0)

    # prepare data
    data = prepare_optimization_data(
        huc8_df, solar_proportion_df, wind_proportion_df, demand_profile, train_path=config['train_path']
    )

    # prepare weights
    weights_df = pd.read_csv(config['weights_file'], index_col=0)

    new_weights = []

    for idx, weights in weights_df.iterrows():
        # only run optimization problems that haven't been computed yet
        if os.path.exists(f"Data/Models/{config['scenario_name']}/{weights['weights_name']}_alpha_{weights['alpha']}_beta_{weights['beta']}_gamma_{weights['gamma']}_delta_{weights['delta']}.pkl"):
            pass
        else:
            new_weights.append(idx)

    new_weights_df = weights_df.loc[new_weights]

    weights_dict = new_weights_df.T.to_dict()

    # run the optimization
    optimize_data_center_siting(
        data, 
        scenario_name=config['scenario_name'],
        weights_dict=weights_dict,
        equity_type=config['equity_type'],
        verbose=True,
        grid_only=config['grid_only'],
        normalization=config['normalization'],
        train_path=config['train_path']
    )


if __name__ == "__main__":
    print("Data Center Siting Optimization Module Loaded")
    print("Use run_optimization() function to execute the optimization")
    print("\nExample:")
    print("results = run_optimization(config_file)")