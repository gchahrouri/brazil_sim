import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import logging
import itertools

logging.basicConfig(level=logging.DEBUG)

def apply_stochastic_variation(base_value, variation_scale, randomness_scale, distribution='normal'):
    if distribution == 'normal':
        return np.clip(np.random.normal(base_value, variation_scale * base_value * randomness_scale), 
                       0.1 * base_value, 2 * base_value)  # Clip between 10% and 200% of base value
    elif distribution == 'lognormal':
        target_mean = base_value
        target_std = base_value * variation_scale * randomness_scale
        sigma = np.sqrt(np.log(1 + (target_std**2 / target_mean**2)))
        mu = np.log(target_mean**2 / np.sqrt(target_std**2 + target_mean**2))
        return np.clip(np.random.lognormal(mu, sigma), 
                       0.1 * base_value, 2 * base_value)  # Clip between 10% and 200% of base value
    else:
        raise ValueError("Distribution must be either 'normal' or 'lognormal'")

def calculate_adjusted_growth_rates(target_cagr, years, market_decay_rate):
    def objective(initial_rate):
        rates = initial_rate * np.exp(-market_decay_rate * np.arange(years))
        overall_growth = np.prod(1 + rates) - 1
        return (overall_growth - (1 + target_cagr)**years + 1)**2

    result = minimize_scalar(objective)
    optimal_initial_rate = result.x
    growth_rates = optimal_initial_rate * np.exp(-market_decay_rate * np.arange(years))
    return growth_rates

def calculate_marketing_effectiveness(marketing_spend, marketing_roi, scale_growth_factor, market_share):
    return marketing_spend * marketing_roi * (scale_growth_factor * (1 + market_share))

def update_market_size(previous_size, growth_rate, randomness_scale):
    return previous_size * (1 + apply_stochastic_variation(growth_rate, 0.05, randomness_scale, 'normal'))

def adjust_retention_rate(base_rate, year, start_year, min_rate_factor, max_rate_factor, retention_decay_rate):
    years_passed = year - start_year
    adjustment = 1 - np.exp(-retention_decay_rate * years_passed)
    min_rate = min_rate_factor * base_rate
    max_rate = max_rate_factor * base_rate
    adjusted_rate = min_rate + (max_rate - min_rate) * adjustment
    return adjusted_rate

def update_competitor_metrics(competitors, randomness_scale, current_year, start_year, min_rate_factor, max_rate_factor, retention_decay_rate):
    for competitor in competitors:
        competitor['marketing_spend'] *= (1 + apply_stochastic_variation(
            competitor['spend_growth'], 0.05, randomness_scale, 'normal'))
        
        new_marketing_roi = max(0, apply_stochastic_variation(
            competitor['marketing_roi'], 0.05, randomness_scale, 'lognormal'))
        if new_marketing_roi < 0.1:  # Flag if marketing ROI is less than 0.1
            logging.warning(f"Low marketing ROI detected for {competitor['name']}: {new_marketing_roi}")
        competitor['marketing_roi'] = new_marketing_roi
        
        base_retention_rate = competitor['base_retention_rate']
        adjusted_retention_rate = adjust_retention_rate(base_retention_rate, current_year, start_year,
                                                       min_rate_factor, max_rate_factor, retention_decay_rate)
        new_retention_rate = max(0, min(1, apply_stochastic_variation(
            adjusted_retention_rate, 0.05, randomness_scale, 'normal')))
        if new_retention_rate < 0.1:  # Flag if retention rate is less than 0.1
            logging.warning(f"Low retention rate detected for {competitor['name']}: {new_retention_rate}")
        competitor['retention_rate'] = new_retention_rate
    
    return competitors

def run_simulation(market_size_2025, market_cagr, market_decay_rate,
                   competitors, simulation_years, randomness_scale,
                   min_rate_factor, max_rate_factor, retention_decay_rate, num_simulations):
    logging.debug(f"Starting simulation with {num_simulations} iterations")
    logging.debug(f"Initial market size: {market_size_2025}, CAGR: {market_cagr}, Decay rate: {market_decay_rate}")
    logging.debug(f"Competitors: {competitors}")
    
    all_results = []
    for sim in range(num_simulations):
        growth_rates = calculate_adjusted_growth_rates(market_cagr, simulation_years, market_decay_rate)
        market_size = apply_stochastic_variation(market_size_2025 * 1e9, 0.05, randomness_scale, 'normal')
        years = np.arange(2025, 2025 + simulation_years)
        
        sim_competitors = [competitor.copy() for competitor in competitors]
        for competitor in sim_competitors:
            competitor['revenue'] = market_size * competitor['market_share']
            competitor['base_retention_rate'] = competitor['retention_rate']

        for year_idx in range(simulation_years):
            current_year = years[year_idx]
            market_size = update_market_size(market_size, growth_rates[year_idx], randomness_scale)
            sim_competitors = update_competitor_metrics(sim_competitors, randomness_scale, current_year, years[0],
                                                        min_rate_factor, max_rate_factor, retention_decay_rate)
            
            retained_revenues = np.array([comp['revenue'] * comp['retention_rate'] for comp in sim_competitors])
            total_available_revenue = max(market_size - np.sum(retained_revenues), 0)
            
            marketing_effectiveness = np.array([
                calculate_marketing_effectiveness(
                    comp['marketing_spend'], comp['marketing_roi'],
                    comp['scale_growth_factor'], comp['market_share']
                ) for comp in sim_competitors
            ])
            total_effectiveness = np.sum(marketing_effectiveness)
            new_revenues = (marketing_effectiveness / total_effectiveness) * total_available_revenue if total_effectiveness > 0 else np.zeros_like(marketing_effectiveness)

            total_revenue = sum(retained_revenues) + sum(new_revenues)
            for idx, competitor in enumerate(sim_competitors):
                competitor['revenue'] = retained_revenues[idx] + new_revenues[idx]
                competitor['market_share'] = competitor['revenue'] / total_revenue
                all_results.append({
                    'simulation': sim,
                    'year': current_year,
                    'competitor': competitor['name'],
                    'revenue': competitor['revenue'],
                    'market_share': competitor['market_share'],
                    'retention_rate': competitor['retention_rate'],
                    'marketing_roi': competitor['marketing_roi']
                })

    result_df = pd.DataFrame(all_results)
    
    # Add summary statistics
    logging.debug("Summary statistics for marketing ROI:")
    logging.debug(result_df.groupby('competitor')['marketing_roi'].describe())
    
    logging.debug("Summary statistics for retention rate:")
    logging.debug(result_df.groupby('competitor')['retention_rate'].describe())
    
    logging.debug(f"Simulation results shape: {result_df.shape}")
    logging.debug(f"Simulation results head: {result_df.head()}")
    return result_df

def run_sensitivity_analysis(params, superbet_params, sensitivity_params, competitors, steps, variation):
    logging.debug(f"Starting sensitivity analysis for parameters: {sensitivity_params}")
    logging.debug(f"Params: {params}")
    logging.debug(f"Superbet params: {superbet_params}")
    logging.debug(f"Competitors: {competitors}")
    logging.debug(f"Steps: {steps}, Variation: {variation}")
    
    results = []
    param_combinations = []

    if len(sensitivity_params) == 2:
        param1, param2 = sensitivity_params
        base_value1 = superbet_params[param1]
        base_value2 = superbet_params[param2]
        param_values1 = np.linspace(base_value1 * (1 - variation), base_value1 * (1 + variation), steps)
        param_values2 = np.linspace(base_value2 * (1 - variation), base_value2 * (1 + variation), steps)
        param_combinations = list(itertools.product(param_values1, param_values2))
    else:
        for param in sensitivity_params:
            base_value = superbet_params[param]
            param_values = np.linspace(base_value * (1 - variation), base_value * (1 + variation), steps)
            param_combinations.append([(param, value) for value in param_values])
        param_combinations = list(itertools.product(*param_combinations))

    for combo in param_combinations:
        temp_competitors = [comp.copy() for comp in competitors]
        superbet = next(comp for comp in temp_competitors if comp['name'] == 'Superbet')
        
        if len(sensitivity_params) == 2:
            superbet[sensitivity_params[0]] = combo[0]
            superbet[sensitivity_params[1]] = combo[1]
        else:
            for param, value in combo:
                superbet[param] = value

        sim_results = run_simulation(
            params['market_size_2025'], params['market_cagr'], params['market_decay_rate'],
            temp_competitors, 6, params['randomness_scale'],
            params['min_retention_factor'], params['max_retention_factor'],
            params['retention_decay_rate'], params['num_simulations']
        )

        summary = sim_results[sim_results['competitor'] == 'Superbet'].groupby('year').agg({
            'market_share': ['mean', 'std']
        }).reset_index()
        summary.columns = ['year', 'market_share_mean', 'market_share_std']

        for _, row in summary.iterrows():
            result = {
                'year': row['year'],
                'market_share_mean': row['market_share_mean'],
                'market_share_std': row['market_share_std']
            }
            if len(sensitivity_params) == 2:
                result[sensitivity_params[0]] = combo[0]
                result[sensitivity_params[1]] = combo[1]
            else:
                for param, value in combo:
                    result[param] = value
            results.append(result)

    result_df = pd.DataFrame(results)
    
    logging.debug(f"Sensitivity analysis results shape: {result_df.shape}")
    logging.debug(f"Sensitivity analysis results head: {result_df.head()}")
    logging.debug(f"Sensitivity analysis results columns: {result_df.columns}")
    return result_df
