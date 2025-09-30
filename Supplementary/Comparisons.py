import os
import pickle
import configparser
from typing import Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import cvxpy as cp
import wandb
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse
from pyproj import Transformer

# ============================================================================
# config and data paths
# ============================================================================

BASE_PATH = ...

DATA_PATHS = {
    'solar': os.path.join(BASE_PATH, "Solar/solar_proportion_huc8.csv"),
    'wind': os.path.join(BASE_PATH, "Wind/wind_proportion_huc8.csv"),
    'flat_demand': os.path.join(BASE_PATH, "Demand Profiles/flat_2_GWh.csv"),
    'business_demand': os.path.join(BASE_PATH, "Demand Profiles/business_2GWh.csv"),
    'caiso_demand': os.path.join(BASE_PATH, "Demand Profiles/caiso_2013_2GWh.csv"),
    'huc8_geojson': os.path.join(BASE_PATH, "Footprint/california.geojson"),
    'models_dir': os.path.join(BASE_PATH, "Models"),
    'figures_dir': os.path.join(BASE_PATH, "Figures"),
    'configs_dir': os.path.join(BASE_PATH, "Configs")
}

# color scheme for the visualizations
COLOR_SCHEME = {
    'Grid': 'silver',
    'Solar': 'yellow',
    'Wind': 'deepskyblue',
    'Data Center': 'red',
    'Curtailed Solar': 'yellow',
    'Curtailed Wind': 'deepskyblue'
}

# ============================================================================
# data prep
# ============================================================================

def load_data():
    data = {}
    data['solar_proportion'] = pd.read_csv(DATA_PATHS['solar'], index_col=0)
    data['wind_proportion'] = pd.read_csv(DATA_PATHS['wind'], index_col=0)
    data['flat_demand'] = pd.read_csv(DATA_PATHS['flat_demand'], index_col=0)
    data['business_demand'] = pd.read_csv(DATA_PATHS['business_demand'], index_col=0)
    data['caiso_demand'] = pd.read_csv(DATA_PATHS['caiso_demand'], index_col=0)
    data['huc8_df'] = gpd.read_file(DATA_PATHS['huc8_geojson'])
    return data

def prepare_optimization_data(huc8_df, solar_proportion_df, wind_proportion_df,
                             demand_profile, data_center_cost=6e5):
    """prep all data matrices for the optimization"""
    
    L = len(huc8_df)
    T = len(demand_profile)
    
    huc8_order = huc8_df['HUC8'].values
    
    # cost data
    P_dc = np.ones(L) * data_center_cost
    P_g = huc8_df['Electricity Price [$/MWh]'].values
    P_s = huc8_df['Mean Solar LCOE [$/MWh]'].values
    P_w = huc8_df['Mean Wind LCOE [$/MWh]'].values
    
    footprint_ordered = huc8_df.set_index('HUC8_str').reindex(huc8_order).reset_index()
    
    # water scarcity footprint data
    S_dc = footprint_ordered['Data Center Water Scarcity Footprint [m3-eq/MWh]'].values
    S_g = footprint_ordered['Grid Water Scarcity Footprint [m3-eq/MWh]'].values
    S_s = footprint_ordered['Solar Water Scarcity Footprint [m3-eq/MWh]'].values
    S_w = footprint_ordered['Wind Water Scarcity Footprint [m3-eq/MWh]'].values
    
    # carbon footprint data
    E_g = footprint_ordered['Grid Carbon Footprint [tons CO2-eq/MWh]'].values
    E_s = footprint_ordered['Solar Carbon Footprint [tons CO2-eq/MWh]'].values
    E_w = footprint_ordered['Wind Carbon Footprint [tons CO2-eq/MWh]'].values
    
    # solar and wind proportion data
    C_s = solar_proportion_df[huc8_order].values.T
    C_w = wind_proportion_df[huc8_order].values.T
    
    # normalize to proportions
    C_s = C_s / (C_s.sum(axis=1, keepdims=True) + 1e-10)
    C_w = C_w / (C_w.sum(axis=1, keepdims=True) + 1e-10)
    
    D = demand_profile.values.flatten()    
    Y = np.zeros((L, T))
    
    return {
        'L': L, 'T': T,
        'P_dc': P_dc, 'P_g': P_g, 'P_s': P_s, 'P_w': P_w,
        'S_dc': S_dc, 'S_g': S_g, 'S_s': S_s, 'S_w': S_w,
        'E_g': E_g, 'E_s': E_s, 'E_w': E_w,
        'C_s': C_s, 'C_w': C_w,
        'D': D, 'Y': Y,
        'huc8_order': huc8_order
    }

# ============================================================================
# optimization
# ============================================================================

def compute_composite_costs(data, alpha, beta, gamma, normalization='grid_std'):
    """compute the composite cost matrices M_g, M_s, M_w"""
    
    all_S = np.concatenate([data['S_g'], data['S_s'], data['S_w'], data['S_dc']])
    all_P = np.concatenate([data['P_g'], data['P_s'], data['P_w'], 
                           data['P_dc']/8760])  # Convert $/MW-year to $/MWh
    all_E = np.concatenate([data['E_g'], data['E_s'], data['E_w'], 
                           np.zeros(data['E_g'].shape[0])])
    
    # compute normalization factors based on method
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
        raise ValueError(f"Unrecognized normalization {normalization}")
    
    # avoid division by zero
    norm_S = max(norm_S, 1e-10)
    norm_P = max(norm_P, 1e-10)
    norm_E = max(norm_E, 1e-10)
    
    # compute composite costs
    M_g = alpha * (data['S_g'] / norm_S) + beta * (data['P_g'] / norm_P) + gamma * (data['E_g'] / norm_E)
    M_s = alpha * (data['S_s'] / norm_S) + beta * (data['P_s'] / norm_P) + gamma * (data['E_s'] / norm_E)
    M_w = alpha * (data['S_w'] / norm_S) + beta * (data['P_w'] / norm_P) + gamma * (data['E_w'] / norm_E)
    
    return M_g, M_s, M_w, norm_S, norm_P, norm_E

def optimize_data_center_siting(data, scenario_name, weights_dict, 
                                equity_type='max', normalization='grid_std',
                                verbose=True, grid_only=False):
    
    # create params
    alpha = cp.Parameter(nonneg=True)
    beta = cp.Parameter(nonneg=True)
    gamma = cp.Parameter(nonneg=True)
    delta = cp.Parameter(nonneg=True)
    
    L, T = data['L'], data['T']
    
    M_g, M_s, M_w, norm_S, norm_P, norm_E = compute_composite_costs(
        data, alpha, beta, gamma, normalization)
    
    # decision vars
    x = cp.Variable((L, 1), nonneg=True)  # New DC capacity [MW]
    a = cp.Variable((L, T), nonneg=True)  # DC demand allocation [MWh]
    g = cp.Variable((L, T), nonneg=True)  # Grid power [MWh]
    s = cp.Variable((L, 1), nonneg=True)  # Annual solar [MWh]
    w = cp.Variable((L, 1), nonneg=True)  # Annual wind [MWh]
    
    # water scarcity vector
    S = (cp.diag(data['S_g']) @ cp.sum(g, axis=1, keepdims=True) +
         cp.diag(data['S_s']) @ s +
         cp.diag(data['S_w']) @ w +
         cp.diag(data['S_dc']) @ cp.sum(a, axis=1, keepdims=True))
    
    # water inequity term
    if equity_type == 'max':
        f_equity = cp.max(S)
        equity_constraints = []
    elif equity_type == 'mad':
        f_equity = cp.sum(cp.abs(S - S.T)) / (L * L)
        equity_constraints = []
    else:
        raise ValueError(f"Equity type {equity_type} not recognized")
    
    constraints = []
    constraints.append(cp.sum(a, axis=0) >= data['D'])  # Meet demand
    constraints.append(g + cp.diag(s.flatten()) @ data['C_s'] + 
                      cp.diag(w.flatten()) @ data['C_w'] >= a)  # Power balance
    constraints.append(x + data['Y'] >= a)  # Capacity constraint
    constraints.extend(equity_constraints)
    
    if grid_only:
        for l in range(L):
            constraints.append(s[l] == 0)
            constraints.append(w[l] == 0)
    
    # obj function
    obj = ((beta / norm_P) * (data['P_dc'].T @ x) +
           M_g.T @ cp.sum(g, axis=1) +
           M_s.T @ s +
           M_w.T @ w +
           (alpha / norm_S) * (data['S_dc'].T @ cp.sum(a, axis=1)) +
           (delta / norm_S) * f_equity)
    
    problem = cp.Problem(cp.Minimize(obj), constraints)
    
    results = {}
    
    for weight_name, params in weights_dict.items():
        alpha.value = params['alpha']
        beta.value = params['beta']
        gamma.value = params['gamma']
        delta.value = params['delta']
        
        problem.solve(solver='GUROBI', verbose=verbose, warm_start=True,
                     Threads=8, Presolve=2, Method=2, Crossover=0,
                     BarConvTol=1e-7, FeasibilityTol=1e-7, OptimalityTol=1e-7,
                     NumericFocus=1)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            print(f"Warning: Problem status is {problem.status}")
        
        # extract results
        result = {
            'x': x.value, 'a': a.value, 'g': g.value,
            's': s.value, 'w': w.value, 'S': S.value,
            'Water_Inequity': f_equity.value,
            'Objective_Value': problem.value,
            'status': problem.status,
            'scenario_name': weight_name,
            'normalization': normalization
        }
        result.update(params)
        
        # analyze 
        results_df, total_metrics = analyze_results(result, data)
        
        # inequality metrics
        ineq_metrics = compute_inequality_metrics(result, data)
        total_metrics.update(ineq_metrics)
        
        if verbose:
            print(f"\n{weight_name} - {normalization}:")
            print(f"  Total New Capacity: {total_metrics['Total_New_Capacity_MW']:.1f} MW")
            print(f"  Renewable Energy: {total_metrics['Renewable_Percent']:.1f}%")
            print(f"  Max Water Scarcity: {total_metrics['max_water_scarcity']:.0f} m³-eq")
            print(f"  Theil Index: {total_metrics['theil_index']:.4f}")
        
        
        save_path = os.path.join(DATA_PATHS['models_dir'], 
                                 f"{scenario_name}_{normalization}",
                                 f"{weight_name}_alpha_{params['alpha']}_beta_{params['beta']}_"
                                 f"gamma_{params['gamma']}_delta_{params['delta']}.pkl")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, "wb") as f:
            pickle.dump({'results': result, 'results_df': results_df, 
                        'metrics': total_metrics}, f)
        
        results[weight_name] = {
            'results': result,
            'results_df': results_df,
            'metrics': total_metrics
        }
    
    return results

# ============================================================================
# post analysis
# ============================================================================

def analyze_results(results, data):
    
    results_df = pd.DataFrame({
        'HUC8': data['huc8_order'],
        'New_Capacity_MW': results['x'].flatten(),
        'Total_Grid_MWh': np.sum(results['g'], axis=1),
        'Solar_MWh': results['s'].flatten(),
        'Wind_MWh': results['w'].flatten(),
        'Total_Demand_MWh': np.sum(results['a'], axis=1),
        'Data_Center_Cost_Per_MWh': data['P_dc']
    })
    
    total_energy = (results_df['Total_Grid_MWh'] + 
                   results_df['Solar_MWh'] + 
                   results_df['Wind_MWh'])
    
    results_df['Grid_Percent'] = 100 * results_df['Total_Grid_MWh'] / (total_energy + 1e-10)
    results_df['Solar_Percent'] = 100 * results_df['Solar_MWh'] / (total_energy + 1e-10)
    results_df['Wind_Percent'] = 100 * results_df['Wind_MWh'] / (total_energy + 1e-10)
    results_df['Data_Center_Capacity_Factor'] = (
        results_df['Total_Demand_MWh'].where(results_df['Total_Demand_MWh'] > 1, 0) / 
        (results_df['New_Capacity_MW'] * 8760 + 1e-10)
    )
    
    # calc total metrics
    total_metrics = {
        'Total_New_Capacity_MW': np.sum(results['x']),
        'Total_Grid_MWh': np.sum(results['g']),
        'Total_Solar_MWh': np.sum(results['s']),
        'Total_Wind_MWh': np.sum(results['w']),
        'Water_Inequity': results['Water_Inequity'],
        'Objective_Value': results['Objective_Value']
    }
    
    # environmental impacts
    total_emissions = (data['E_g'].T @ np.sum(results['g'], axis=1) +
                      data['E_s'].T @ results['s'].flatten() +
                      data['E_w'].T @ results['w'].flatten())
    
    total_water = (data['S_g'].T @ np.sum(results['g'], axis=1) +
                  data['S_s'].T @ results['s'].flatten() +
                  data['S_w'].T @ results['w'].flatten())
    
    total_metrics['Total_Emissions_tonsCO2'] = float(total_emissions)
    total_metrics['Total_Water_Scarcity_m3eq'] = float(total_water)
    
    # renewable percent
    total_energy_all = (total_metrics['Total_Grid_MWh'] + 
                       total_metrics['Total_Solar_MWh'] + 
                       total_metrics['Total_Wind_MWh'])
    total_metrics['Solar_Percent'] = 100 * total_metrics['Total_Solar_MWh'] / total_energy_all
    total_metrics['Wind_Percent'] = 100 * total_metrics['Total_Wind_MWh'] / total_energy_all
    total_metrics['Renewable_Percent'] = total_metrics['Solar_Percent'] + total_metrics['Wind_Percent']
    
    return results_df, total_metrics

def compute_inequality_metrics(results, data):
    
    # water scarcity vector
    S = (data['S_g'] * np.sum(results['g'], axis=1) +
         data['S_s'] * results['s'].flatten() +
         data['S_w'] * results['w'].flatten() +
         data['S_dc'] * np.sum(results['a'], axis=1))
    
    metrics = {
        'max_water_scarcity': np.max(S),
        'mad_water_scarcity': np.mean(np.abs(S[:, np.newaxis] - S)),
        'theil_index': theil_index(S),
        'atkinson_0.5': atkinson_index(S, epsilon=0.5),
        'atkinson_1.0': atkinson_index(S, epsilon=1.0),
        'atkinson_2.0': atkinson_index(S, epsilon=2.0)
    }
    
    return metrics

def theil_index(S, eps=1e-12):
    S = np.asarray(S, dtype=float).reshape(-1)
    S = np.clip(S, 0.0, None)
    m = S.mean()
    if m <= eps:
        return 0.0
    r = S / m
    return float((r * np.log(r + eps)).mean())

def atkinson_index(S, epsilon=1.0, eps=1e-12):
    S = np.asarray(S, dtype=float).reshape(-1)
    S = np.clip(S, 0.0, None)
    m = S.mean()
    
    if m <= eps:
        return 0.0
    elif abs(epsilon - 1.0) < 1e-9:
        g = np.exp(np.log(S + eps).mean())
        return float(1.0 - g / m)
    else:
        p = 1.0 - epsilon
        mean_power = np.mean((S + eps)**p)
        eq = mean_power**(1.0/p)
        return float(1.0 - eq / m)

def curtailed_renewables(data, results):
    """level of used versus curtailed solar and wind"""
    
    solar_hourly = np.diag(results['s'].flatten()) @ data['C_s']
    wind_hourly = np.diag(results['w'].flatten()) @ data['C_w']
    
    total_renewables = solar_hourly + wind_hourly
    renewables_divide = np.where(total_renewables > 0, total_renewables, 1)
    
    solar_used = np.where(total_renewables >= results['a'], 
                         (results['a']/renewables_divide) * solar_hourly, 
                         solar_hourly)
    solar_curtailed = solar_hourly - solar_used
    
    wind_used = np.where(total_renewables >= results['a'],
                        (results['a']/renewables_divide) * wind_hourly,
                        wind_hourly)
    wind_curtailed = wind_hourly - wind_used
    
    return solar_used, solar_curtailed, wind_used, wind_curtailed

# ============================================================================
# visualizations
# ============================================================================

def enhance_results_gdf(huc8_df, results_df, results, data):
    
    solar_used, solar_curtailed, wind_used, wind_curtailed = curtailed_renewables(data, results)
    
    results_gdf = huc8_df.merge(results_df, on="HUC8")
    
    # renewables
    results_gdf['Solar_Used_MWh'] = np.sum(solar_used, axis=1)
    results_gdf['Solar_Curtailed_MWh'] = np.sum(solar_curtailed, axis=1)
    results_gdf['Wind_Used_MWh'] = np.sum(wind_used, axis=1)
    results_gdf['Wind_Curtailed_MWh'] = np.sum(wind_curtailed, axis=1)
    
    # water scarcity
    results_gdf['Total Grid Water Scarcity Footprint [m^3-eq]'] = (
        results_gdf['Total_Grid_MWh'] * results_gdf['Grid Water Scarcity Footprint [m3-eq/MWh]'])
    results_gdf['Total Solar Water Scarcity Footprint [m^3-eq]'] = (
        results_gdf['Solar_MWh'] * results_gdf['Solar Water Scarcity Footprint [m3-eq/MWh]'])
    results_gdf['Total Wind Water Scarcity Footprint [m^3-eq]'] = (
        results_gdf['Wind_MWh'] * results_gdf['Wind Water Scarcity Footprint [m3-eq/MWh]'])
    results_gdf['Total Data Center Water Scarcity Footprint [m^3-eq]'] = (
        results_gdf['Total_Demand_MWh'] * results_gdf['Data Center Water Scarcity Footprint [m3-eq/MWh]'])
    
    # emissions
    results_gdf['Total Grid Emissions [tons CO2-eq]'] = (
        results_gdf['Total_Grid_MWh'] * results_gdf['Grid Carbon Footprint [tons CO2-eq/MWh]'])
    results_gdf['Total Solar Emissions [tons CO2-eq]'] = (
        results_gdf['Solar_MWh'] * results_gdf['Solar Carbon Footprint [tons CO2-eq/MWh]'])
    results_gdf['Total Wind Emissions [tons CO2-eq]'] = (
        results_gdf['Wind_MWh'] * results_gdf['Wind Carbon Footprint [tons CO2-eq/MWh]'])
    
    # costs
    results_gdf['Total Grid Cost [$]'] = (
        results_gdf['Total_Grid_MWh'] * results_gdf['Electricity Price [$/MWh]'])
    results_gdf['Total Solar Cost [$]'] = (
        results_gdf['Solar_MWh'] * results_gdf['Mean Solar LCOE [$/MWh]'])
    results_gdf['Total Wind Cost [$]'] = (
        results_gdf['Wind_MWh'] * results_gdf['Mean Wind LCOE [$/MWh]'])
    results_gdf['Total Data Center Cost [$]'] = (
        results_gdf['New_Capacity_MW'] * results_gdf['Data_Center_Cost_Per_MWh'])
    
    results_gdf.to_crs('EPSG:4326', inplace=True)
    
    return results_gdf

def visualize_stats(df, col, title, cmap, cmap_label, cmap_lims=None, ax=None):
    
    mpl.rcParams.update({'font.size': 20})
    
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    
    if cmap_lims is None:
        df.plot(column=col, edgecolor='black', linewidth=0.5, cmap=cmap, 
               legend=True, legend_kwds={"label": cmap_label}, ax=ax)
    else:
        df.plot(column=col, edgecolor='black', linewidth=0.5, cmap=cmap,
               vmin=cmap_lims[0], vmax=cmap_lims[1],
               legend=True, legend_kwds={"label": cmap_label}, ax=ax)
    
    ax.set_title(title)
    ax.set_axis_off()
    
    return ax

def geoplot_pie(df, lat_col, lon_col, category_dict, unit_factor, unit, 
                pie_scale, ax=None, num_circles=3):
    
    if ax is None:
        ax = plt.gca()
    
    df.plot(ax=ax, edgecolor='black', facecolor='white', linewidth=0.5)
    
    for _, row in df.iterrows():
        if row[category_dict.values()].sum() > 0:
            lon = row[lon_col]
            lat = row[lat_col]
            
            ax.pie(row[category_dict.values()], 
                  radius=np.sqrt(row[category_dict.values()].sum()) * pie_scale,
                  center=(lon, lat), 
                  colors=[COLOR_SCHEME[key] for key in category_dict.keys()],
                  wedgeprops={'edgecolor': 'black', 'linewidth': 1})
    
    ref_sizes = np.linspace(0, df[category_dict.values()].sum(axis=1).max(), 
                           num_circles+1)[1:]
    biggest = np.sqrt(ref_sizes[-1]) * pie_scale
    
    positions = [(-125.5, 35), (-125.5, 33.3), (-125.5, 31.6)]
    for size, (lon, lat) in zip(ref_sizes, positions[3-num_circles:]):
        radius = np.sqrt(size) * pie_scale
        e = Ellipse(xy=(lon, lat), width=radius*2, height=radius*2, 
                   angle=0, alpha=0.5, color='gray')
        ax.add_artist(e)
        ax.text(lon+biggest+0.2, lat-0.2, f"{size * unit_factor:.0f}" + unit)
    
    for key in category_dict.keys():
        ax.bar(0, 0, color=COLOR_SCHEME[key], label=key, zorder=-3)
    
    ax.set_xlim(lon - 1, ax.get_xlim()[1])
    ax.set_ylim(lat - 1, ax.get_ylim()[1] + 0.5)
    ax.legend(loc='upper right', fontsize=15)

def create_comparison_plots(results_dict, huc8_df, data, scenario_name):
    
    fig, axes = plt.subplots(2, len(results_dict), figsize=(7*len(results_dict), 14))
    
    for idx, (norm_method, norm_results) in enumerate(results_dict.items()):
        for weight_name, weight_results in norm_results.items():
            if 'Balanced' in weight_name:  # Example for one weight set
                results_gdf = enhance_results_gdf(
                    huc8_df, 
                    weight_results['results_df'],
                    weight_results['results'],
                    data
                )
                
                # data center capacity
                visualize_stats(results_gdf, 'New_Capacity_MW', 
                              f'Added DC Capacity\n({norm_method})',
                              'Greens', 'MW', ax=axes[0, idx])
                
                # capacity factor
                visualize_stats(results_gdf, 'Data_Center_Capacity_Factor',
                              f'DC Capacity Factor\n({norm_method})',
                              'Blues', '', [0, 1], ax=axes[1, idx])
    
    plt.tight_layout()
    save_path = os.path.join(DATA_PATHS['figures_dir'], 
                             f"{scenario_name}_normalization_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

# ============================================================================
# main
# ============================================================================

def run_comparison_analysis(scenario_name="california_flat_withRenewables_max"):
    
    print("loading data...")
    data_dict = load_data()
    
    # Prepare optimization data
    data = prepare_optimization_data(
        data_dict['huc8_df'],
        data_dict['solar_proportion'],
        data_dict['wind_proportion'],
        data_dict['flat_demand']
    )
    
    # define weight scenarios
    weights_dict = {
        'Cost_Only': {'weights_name': 'Cost_Only', 'alpha': 0, 'beta': 1, 'gamma': 0, 'delta': 0},
        'Balanced': {'weights_name': 'Balanced', 'alpha': 1, 'beta': 1, 'gamma': 1, 'delta': 0},
        'With_Equity': {'weights_name': 'With_Equity', 'alpha': 1, 'beta': 1, 'gamma': 1, 'delta': 1},
        'Strong_Equity': {'weights_name': 'Strong_Equity', 'alpha': 1, 'beta': 1, 'gamma': 1, 'delta': 5}
    }
    
    all_results = {}
    
    # optimization with both normalization methods
    for normalization in ['all_max', 'grid_std']:
        print(f"\n{'='*60}")
        print(f"running the optimization with {normalization} normalization")
        print(f"{'='*60}")
        
        results = optimize_data_center_siting(
            data=data,
            scenario_name=scenario_name,
            weights_dict=weights_dict,
            equity_type='max',
            normalization=normalization,
            verbose=True,
            grid_only=False
        )
        
        all_results[normalization] = results
    
    # create comparison visualizations
    print("\ncreating comparison plots...")
    create_comparison_plots(all_results, data_dict['huc8_df'], data, scenario_name)
    
    # analysis 
    for norm_method, norm_results in all_results.items():
        print(f"\n{'='*60}")
        print(f"Results summary for {norm_method} normalization")
        print(f"{'='*60}")
        
        for weight_name, weight_results in norm_results.items():
            metrics = weight_results['metrics']
            print(f"\n{weight_name}:")
            print(f"  Total Capacity: {metrics['Total_New_Capacity_MW']:.1f} MW")
            print(f"  Renewable %: {metrics['Renewable_Percent']:.1f}%")
            print(f"  Emissions: {metrics['Total_Emissions_tonsCO2']:.0f} tons CO2")
            print(f"  Water Scarcity: {metrics['Total_Water_Scarcity_m3eq']:.0f} m³-eq")
            print(f"  Max Water Scarcity: {metrics['max_water_scarcity']:.0f} m³-eq")
            print(f"  Theil Index: {metrics['theil_index']:.4f}")
            print(f"  Atkinson (ε=1.0): {metrics['atkinson_1.0']:.4f}")
    
    return all_results

def create_comprehensive_visualizations(results_dict, huc8_df, data, scenario_name):
    
    usage_dict = {
        'Grid': 'Total_Grid_MWh',
        'Solar': 'Solar_Used_MWh',
        'Wind': 'Wind_Used_MWh'
    }
    
    curtail_dict = {
        'Solar': 'Solar_Curtailed_MWh',
        'Wind': 'Wind_Curtailed_MWh'
    }
    
    water_dict = {
        'Grid': 'Total Grid Water Scarcity Footprint [m^3-eq]',
        'Solar': 'Total Solar Water Scarcity Footprint [m^3-eq]',
        'Wind': 'Total Wind Water Scarcity Footprint [m^3-eq]',
        'Data Center': 'Total Data Center Water Scarcity Footprint [m^3-eq]'
    }
    
    emissions_dict = {
        'Grid': 'Total Grid Emissions [tons CO2-eq]',
        'Solar': 'Total Solar Emissions [tons CO2-eq]',
        'Wind': 'Total Wind Emissions [tons CO2-eq]'
    }
    
    cost_dict = {
        'Grid': 'Total Grid Cost [$]',
        'Solar': 'Total Solar Cost [$]',
        'Wind': 'Total Wind Cost [$]',
        'Data Center': 'Total Data Center Cost [$]'
    }
    
    for weight_name in ['Cost_Only', 'Balanced', 'With_Equity', 'Strong_Equity']:
        fig, axes = plt.subplots(3, 2, figsize=(14, 21))
        fig.suptitle(f'{weight_name} Scenario Comparison', fontsize=16, y=1.02)
        
        for idx, norm_method in enumerate(['all_max', 'grid_std']):
            if weight_name in results_dict[norm_method]:
                weight_results = results_dict[norm_method][weight_name]
                results_gdf = enhance_results_gdf(
                    huc8_df,
                    weight_results['results_df'],
                    weight_results['results'],
                    data
                )
                
                # Row 1: data center capacity
                visualize_stats(results_gdf, 'New_Capacity_MW',
                              f'DC Capacity ({norm_method})',
                              'Greens', 'MW', ax=axes[0, idx])
                
                # Row 2: electricity usage pie chart
                geoplot_pie(results_gdf, "centroid_lat", "centroid_lon",
                           usage_dict, 1e-6, " TWh", 2e-4, ax=axes[1, idx])
                axes[1, idx].set_title(f'Electricity Usage ({norm_method})')
                
                # Row 3: water scarcity footprint pie chart
                geoplot_pie(results_gdf, "centroid_lat", "centroid_lon",
                           water_dict, 1e-6, r'$\times 10^{6}$ m$^3$-eq', 
                           1e-4, ax=axes[2, idx])
                axes[2, idx].set_title(f'Water Scarcity ({norm_method})')
        
        plt.tight_layout()
        save_path = os.path.join(DATA_PATHS['figures_dir'], 
                                 f"{scenario_name}_{weight_name}_comparison.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()

def create_summary_bar_charts(results_dict, huc8_df, data, scenario_name):
    
    metrics_to_plot = [
        ('Total_New_Capacity_MW', 'Total Capacity (MW)', 1),
        ('Renewable_Percent', 'Renewable Energy (%)', 1),
        ('Total_Emissions_tonsCO2', 'Total Emissions (Mt CO2)', 1e-6),
        ('Total_Water_Scarcity_m3eq', 'Total Water Scarcity (M m³-eq)', 1e-6),
        ('theil_index', 'Theil Index', 1),
        ('atkinson_1.0', 'Atkinson Index (ε=1.0)', 1)
    ]
    
    weight_names = ['Cost_Only', 'Balanced', 'With_Equity', 'Strong_Equity']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (metric, label, scale) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        x = np.arange(len(weight_names))
        width = 0.35
        
        all_max_values = [results_dict['all_max'][w]['metrics'][metric] * scale 
                         for w in weight_names]
        grid_std_values = [results_dict['grid_std'][w]['metrics'][metric] * scale 
                          for w in weight_names]
        
        # plot bars
        ax.bar(x - width/2, all_max_values, width, label='all_max', color='steelblue')
        ax.bar(x + width/2, grid_std_values, width, label='grid_std', color='coral')
        
        ax.set_xlabel('Scenario')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(weight_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Normalization Method Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(DATA_PATHS['figures_dir'],
                             f"{scenario_name}_metrics_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

def yearly_heatmap(hourly_np, title, cmap, cmap_label, ax=None):
    
    if ax is None:
        ax = plt.gca()
    
    fig = ax.get_figure()
    hourly_np_heatmap = hourly_np.reshape(365, 24).T
    
    im = ax.imshow(hourly_np_heatmap, aspect=4, cmap=cmap)
    cbar = fig.colorbar(im, orientation='horizontal', ax=ax)
    cbar.set_label(cmap_label)
    
    # set time labels
    time_index = pd.date_range('2013-01-01', periods=8760, freq='h')
    heatmap_ylabels = time_index[:24][::6].strftime('%I%p')
    
    date_list = [time for time in time_index if (time.hour == 0) & (time.day == 1)]
    month_positions = [int(date.strftime('%j'))-1 for date in date_list]
    heatmap_xlabels = [date.strftime('%b') for date in date_list]
    
    ax.set_xticks(month_positions[::3])
    ax.set_xticklabels(heatmap_xlabels[::3])
    ax.set_ylabel("Hour (UTC)")
    ax.set_yticks([0, 6, 12, 18])
    ax.set_yticklabels(heatmap_ylabels)
    ax.set_title(title)

def create_temporal_analysis(results_dict, data, scenario_name):
    
    for norm_method in ['all_max', 'grid_std']:
        for weight_name in ['Cost_Only', 'Balanced', 'Strong_Equity']:
            if weight_name in results_dict[norm_method]:
                weight_results = results_dict[norm_method][weight_name]
                
                solar_used, solar_curtailed, wind_used, wind_curtailed = \
                    curtailed_renewables(data, weight_results['results'])
                
                solar_produced = np.sum(solar_used + solar_curtailed, axis=0)
                wind_produced = np.sum(wind_used + wind_curtailed, axis=0)
                
                fig, axes = plt.subplots(1, 2, figsize=(20, 6))
                
                # total renewables
                yearly_heatmap(solar_produced + wind_produced,
                             f"Total Renewables - {weight_name} ({norm_method})",
                             "Purples", "MWh", ax=axes[0])
                
                # grid electricity
                grid_usage = (data['D'] - np.sum(solar_used + wind_used, axis=0))
                yearly_heatmap(grid_usage,
                             f"Grid Usage - {weight_name} ({norm_method})",
                             "Greys", "MWh", ax=axes[1])
                
                plt.tight_layout()
                save_path = os.path.join(DATA_PATHS['figures_dir'],
                                       f"{scenario_name}_{weight_name}_{norm_method}_temporal.png")
                plt.savefig(save_path, dpi=200, bbox_inches='tight')
                plt.show()



if __name__ == "__main__":
    results = run_comparison_analysis()
    
    data_dict = load_data()
    data = prepare_optimization_data(
        data_dict['huc8_df'],
        data_dict['solar_proportion'],
        data_dict['wind_proportion'],
        data_dict['flat_demand']
    )
    
    print("\ncreating visualizations...")
    create_comprehensive_visualizations(results, data_dict['huc8_df'], data,
                                       "california_flat_withRenewables_max")
    
    print("\ncreating summary bar charts...")
    create_summary_bar_charts(results, data_dict['huc8_df'], data,
                             "california_flat_withRenewables_max")
    
    print("\ncreating temporal analysis...")
    create_temporal_analysis(results, data, "california_flat_withRenewables_max")
    
    print("\n" + "="*60)
    print("analysis done")
    print("="*60)
