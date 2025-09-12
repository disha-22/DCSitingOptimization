import cvxpy as cp
import numpy as np
import pandas as pd
import geopandas as gpd

def prepare_optimization_data(huc8_df, solar_proportion_df, wind_proportion_df,
                            demand_profile, data_center_cost=6e5):
    # as a note, previously data_center_cost was 12e6. Changed to 6e5, to equally spread data center 
    # cost across an estimated lifespan of 20 years. But we can also change it back if necessary
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
    """

    L = len(huc8_df)
    T = len(demand_profile)

    # # detect correct HUC8 column name
    # huc8_col = None
    # for col in ['HUC8', 'huc8', 'HUC_8', 'HUC8_str']:
    #     if col in huc8_df.columns:
    #         huc8_col = col
    #         break

    # fix an order in which to count the HUC8 regions
    huc8_order = huc8_df['HUC8'].values

    # Prepare cost data
    P_dc = np.ones(L) * data_center_cost  # Cost of new data center capacity [$/MW]

    if 'Electricity Price [$/MWh]' in huc8_df.columns:
        P_g = huc8_df['Electricity Price [$/MWh]'].values  # Grid electricity cost [$/MWh]

    P_s = huc8_df['Mean Solar LCOE [$/MWh]'].values  # Solar LCOE [$/MWh]
    P_w = huc8_df['Mean Wind LCOE [$/MWh]'].values  # Wind LCOE [$/MWh]

    # Prepare water scarcity footprint data

    if 'HUC8_str' not in huc8_df.columns:
        huc8_df['HUC8_str'] = huc8_df['HUC8'].map(lambda x: ''.join(['0']*(8-len(str(x)))) + str(x))

    footprint_ordered = huc8_df.set_index('HUC8_str').reindex(huc8_order).reset_index()

    S_dc = footprint_ordered['Data Center Water Scarcity Footprint [m3-eq/MWh]'].values  # DC water scarcity [m³-eq/MWh]
    S_g = footprint_ordered['Grid Water Scarcity Footprint [m3-eq/MWh]'].values  # Grid water scarcity [m³-eq/MWh]
    S_s = footprint_ordered['Solar Water Scarcity Footprint [m3-eq/MWh]'].values  # Solar water scarcity [m³-eq/MWh]
    S_w = footprint_ordered['Wind Water Scarcity Footprint [m3-eq/MWh]'].values  # Wind water scarcity [m³-eq/MWh]

    # Prepare carbon footprint data
    E_g = footprint_ordered['Grid Carbon Footprint [tons CO2-eq/MWh]'].values  # Grid emissions [tons CO2-eq/MWh]
    E_s = footprint_ordered['Solar Carbon Footprint [tons CO2-eq/MWh]'].values  # Solar emissions [tons CO2-eq/MWh]
    E_w = footprint_ordered['Wind Carbon Footprint [tons CO2-eq/MWh]'].values  # Wind emissions [tons CO2-eq/MWh]

    # C_s = np.zeros((L, T))
    # C_w = np.zeros((L, T))

    # # detect HUC8 column for solar 
    # solar_huc8_col = None
    # for col in ['HUC8', 'huc8', 'HUC_8', 'HUC8_str']:
    #     if col in solar_proportion_df.columns:
    #         solar_huc8_col = col
    #         break

    # # detect HUC8 column for wind
    # wind_huc8_col = None
    # for col in ['HUC8', 'huc8', 'HUC_8', 'HUC8_str']:
    #     if col in wind_proportion_df.columns:
    #         wind_huc8_col = col
    #         break

    # for i, huc8 in enumerate(huc8_order):
    #     # find the corresponding data for each HUC8 column
    #     if solar_huc8_col and huc8 in solar_proportion_df[solar_huc8_col].values:
    #         solar_row = solar_proportion_df[solar_proportion_df[solar_huc8_col] == huc8].iloc[0, 1:].values
    #         if len(solar_row) >= T: # truncate time series if longer than time duration
    #             C_s[i, :] = solar_row[:T]
    #         else: # repeat the time series enough times to fill the time duration
    #             repeats = T // len(solar_row) + 1
    #             C_s[i, :] = np.tile(solar_row, repeats)[:T]

    #     if wind_huc8_col and huc8 in wind_proportion_df[wind_huc8_col].values:
    #         wind_row = wind_proportion_df[wind_proportion_df[wind_huc8_col] == huc8].iloc[0, 1:].values
    #         if len(wind_row) >= T:
    #             C_w[i, :] = wind_row[:T]
    #         else:
    #             repeats = T // len(wind_row) + 1
    #             C_w[i, :] = np.tile(wind_row, repeats)[:T]

    # read solar and wind proportion data
    C_s = solar_proportion_df[huc8_order].values.T
    C_w = wind_proportion_df[huc8_order].values.T

    # normalize such that each row sums to 1, and indicates a proportion

    C_s = C_s / (C_s.sum(axis=1, keepdims=True) + 1e-10)
    C_w = C_w / (C_w.sum(axis=1, keepdims=True) + 1e-10)

    # Demand profile
    D = demand_profile.values.flatten()
    # [:T]

    # Existing capacity (assume zero for now, can be updated)
    Y = np.zeros(L)

    return {
        'L': L, 'T': T,
        'P_dc': P_dc, 'P_g': P_g, 'P_s': P_s, 'P_w': P_w,
        'S_dc': S_dc, 'S_g': S_g, 'S_s': S_s, 'S_w': S_w,
        'E_g': E_g, 'E_s': E_s, 'E_w': E_w,
        'C_s': C_s, 'C_w': C_w,
        'D': D, 'Y': Y,
        'huc8_order': huc8_order
    }

def compute_composite_costs(data, alpha=1.0, beta=1.0, gamma=1.0):
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

    Returns
    -------
        M_g: pd.Series
            Composite cost for grid
        M_s: pd.Series
            Composite cost for solar
        M_w: pd.Series
            Composite cost for wind
        sigma_S: float
            Standard deviation for water scarcity footprint
        sigma_P: float
            Standard deviation for monetary cost
        sigma_E: float
            Standard deviation for emissions footprint
    """

    sigma_S = np.std(data['S_g'])
    sigma_P = np.std(data['P_g'])
    sigma_E = np.std(data['E_g'])

    # Avoid division by zero
    sigma_S = max(sigma_S, 1e-10)
    sigma_P = max(sigma_P, 1e-10)
    sigma_E = max(sigma_E, 1e-10)

    # Compute composite costs
    M_g = alpha * (data['S_g'] / sigma_S) + beta * (data['P_g'] / sigma_P) + gamma * (data['E_g'] / sigma_E)
    M_s = alpha * (data['S_s'] / sigma_S) + beta * (data['P_s'] / sigma_P) + gamma * (data['E_s'] / sigma_E)
    M_w = alpha * (data['S_w'] / sigma_S) + beta * (data['P_w'] / sigma_P) + gamma * (data['E_w'] / sigma_E)

    return M_g, M_s, M_w, sigma_S, sigma_P, sigma_E

def optimize_data_center_siting(data, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0,
                               equity_type='max', verbose=True, grid_only=False):
    """
    Solve the data center siting optimization problem

    Parameters:
    -----------
    data : dict
        Dictionary containing all prepared data
    alpha : float
        Weight for water stress
    beta : float
        Weight for monetary costs
    gamma : float
        Weight for emissions
    delta : float
        Weight for water stress inequity
    equity_type : str
        'max' for maximum water scarcity or 'mad' for mean absolute difference
    verbose: boolean
        True if we want to print updates as optimization is in progress.
    grid_only: boolean
        If True, only allow use of grid electricity
    """

    L, T = data['L'], data['T']

    # Compute composite costs
    M_g, M_s, M_w, sigma_S, sigma_P, sigma_E = compute_composite_costs(data, alpha, beta, gamma)

    # Decision variables
    x = cp.Variable((L, 1), nonneg=True)  # New DC capacity [MW]
    a = cp.Variable((L, T), nonneg=True)  # DC demand allocation [MWh]
    g = cp.Variable((L, T), nonneg=True)  # Grid power [MWh]
    s = cp.Variable((L, 1), nonneg=True)  # Annual solar [MWh]
    w = cp.Variable((L, 1), nonneg=True)  # Annual wind [MWh]

    # Compute water scarcity vector S (equation 4)
    S = (cp.diag(data['S_g']) @ cp.sum(g, axis=1, keepdims=True) +
         cp.diag(data['S_s']) @ s +
         cp.diag(data['S_w']) @ w +
         cp.diag(data['S_dc']) @ cp.sum(a, axis=1, keepdims=True))
    # ^ There was previously a "divide by T" on the grid and data center terms. I removed those to fit the optimization model we wrote in Overleaf. - Richard

    print(f"S shape: {S.shape}")
    # Water inequity term
    if equity_type == 'max': # max water scarcity footprint
        f_equity = cp.max(S)
        equity_constraints = []
        # f_equity = cp.Variable(nonneg=True)
        # equity_constraints = [f_equity >= S]
        # equity_constraints = [f_equity >= S[i] for i in range(L)]
    else:  # mean absolute difference of water scarcity footprint
        diff = cp.Variable((L, L), nonneg=True)
        f_equity = cp.sum(diff) / (L * L)
        equity_constraints = []
        equity_constraints.append(diff >= S - S.T)
        equity_constraints.append(diff >= S.T - S)

        # for i in range(L):
        #     for j in range(L):
        #         equity_constraints.append(diff[i, j] >= S[i] - S[j])
        #         equity_constraints.append(diff[i, j] >= S[j] - S[i])

    # Objective function (equation 6)
    obj = ((beta / sigma_P) * (data['P_dc'].T @ x) +
           M_g.T @ cp.sum(g, axis=1) +
           M_s.T @ s +
           M_w.T @ w +
           (alpha / sigma_S) * data['S_dc'].T @ cp.sum(a, axis=1) +
           (delta / sigma_S) * f_equity)

    # Constraints
    constraints = []

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

    # Create and solve problem
    problem = cp.Problem(cp.Minimize(obj), constraints)
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
        'f_equity': f_equity.value,
        'objective': problem.value,
        'status': problem.status
    }

    return results

def analyze_results(results, data, huc8_df):
    """
    Analyze and visualize optimization results
    """
    L, T = data['L'], data['T']

    # Create results dataframe
    results_df = pd.DataFrame({
        'HUC8': data['huc8_order'],
        'New_Capacity_MW': results['x'],
        'Total_Grid_MWh': np.sum(results['g'], axis=1),
        'Solar_MWh': results['s'],
        'Wind_MWh': results['w'],
        'Total_Demand_MWh': np.sum(results['a'], axis=1)
    })

    # Calculate percentages
    total_energy = results_df['Total_Grid_MWh'] + results_df['Solar_MWh'] + results_df['Wind_MWh']
    results_df['Grid_Percent'] = 100 * results_df['Total_Grid_MWh'] / (total_energy + 1e-10)
    results_df['Solar_Percent'] = 100 * results_df['Solar_MWh'] / (total_energy + 1e-10)
    results_df['Wind_Percent'] = 100 * results_df['Wind_MWh'] / (total_energy + 1e-10)

    # Calculate total metrics
    total_metrics = {
        'Total_New_Capacity_MW': np.sum(results['x']),
        'Total_Grid_MWh': np.sum(results['g']),
        'Total_Solar_MWh': np.sum(results['s']),
        'Total_Wind_MWh': np.sum(results['w']),
        'Water_Inequity': results['f_equity'],
        'Objective_Value': results['objective']
    }

    # Calculate environmental impacts
    total_emissions = (np.sum(np.sum(results['g'], axis=1) * data['E_g']) +
                      np.sum(results['s'] * data['E_s']) +
                      np.sum(results['w'] * data['E_w']))

    total_water = (np.sum(np.sum(results['g'], axis=1) * data['S_g']) +
                  np.sum(results['s'] * data['S_s']) +
                  np.sum(results['w'] * data['S_w']) +
                  np.sum(np.sum(results['a'], axis=1) * data['S_dc']))

    total_metrics['Total_Emissions_tonsCO2'] = total_emissions
    total_metrics['Total_Water_Scarcity_m3eq'] = total_water

    # Renewable percentage
    total_energy_all = total_metrics['Total_Grid_MWh'] + total_metrics['Total_Solar_MWh'] + total_metrics['Total_Wind_MWh']
    total_metrics['Renewable_Percent'] = 100 * (total_metrics['Total_Solar_MWh'] + total_metrics['Total_Wind_MWh']) / total_energy_all

    return results_df, total_metrics

def run_optimization(huc8_df, solar_proportion_df, wind_proportion_df,
                    demand_profile, scenario_name="Optimization Results"):
    """
    Run the complete optimization pipeline
    """
    print(f"\n{'='*50}")
    print(f"Running: {scenario_name}")
    print(f"{'='*50}")

    # Prepare data
    data = prepare_optimization_data(
        huc8_df, solar_proportion_df, wind_proportion_df, demand_profile
    )

    # Run optimization with different weight configurations
    scenarios = [
        {'name': 'Cost Only', 'alpha': 0, 'beta': 1, 'gamma': 0, 'delta': 0},
        {'name': 'Balanced', 'alpha': 1, 'beta': 1, 'gamma': 1, 'delta': 0},
        {'name': 'With Equity', 'alpha': 1, 'beta': 1, 'gamma': 1, 'delta': 1},
        {'name': 'Strong Equity', 'alpha': 1, 'beta': 1, 'gamma': 1, 'delta': 5}
    ]

    all_results = {}

    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print("-" * 30)

        results = optimize_data_center_siting(
            data,
            alpha=scenario['alpha'],
            beta=scenario['beta'],
            gamma=scenario['gamma'],
            delta=scenario['delta'],
            verbose=True
        )

        results_df, metrics = analyze_results(results, data, huc8_df)

        print(f"Total New Capacity: {metrics['Total_New_Capacity_MW']:.1f} MW")
        print(f"Renewable Energy: {metrics['Renewable_Percent']:.1f}%")
        print(f"Water Inequity: {metrics['Water_Inequity']:.2f}")
        print(f"Total Emissions: {metrics['Total_Emissions_tonsCO2']:.0f} tons CO2")
        print(f"Total Water Scarcity: {metrics['Total_Water_Scarcity_m3eq']:.0f} m³-eq")

        # Store results
        all_results[scenario['name']] = {
            'results': results,
            'results_df': results_df,
            'metrics': metrics
        }

    return all_results

if __name__ == "__main__":
    print("Data Center Siting Optimization Module Loaded")
    print("Use run_optimization() function to execute the optimization")
    print("\nExample:")
    print("results = run_optimization(huc8_df, solar_proportion_df, wind_proportion_df, flat_demand_2GWh)")