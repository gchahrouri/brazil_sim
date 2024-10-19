##### Chunk 1: Set up the environment
# Please run this chunk first to set up initial parameters
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import seaborn as sns
import ipywidgets as widgets
import itertools
from IPython.display import display, clear_output, HTML
from ipywidgets import VBox, HBox
from scipy.optimize import minimize_scalar

np.random.seed(42)  # For reproducibility
simulation_years = 6  # Number of years to simulate (2025 to 2030)

"""# Simulation Logic"""

##### Chunk 2: Monte Carlo Stochastic Simulation Logic

def apply_stochastic_variation(base_value, variation_scale, randomness_scale, distribution='normal'):
    """
    Apply stochastic variation to a base value using either normal or log-normal distribution.

    Args:
        base_value (float): The initial value to apply variation to.
        variation_scale (float): Scale of the variation relative to the base value.
        randomness_scale (float): Overall scale of randomness in the simulation.
        distribution (str): Type of distribution to use ('normal' or 'lognormal').

    Returns:
        float: The base value with stochastic variation applied.
    """
    if distribution == 'normal':
        return np.random.normal(base_value, variation_scale * base_value * randomness_scale)
    elif distribution == 'lognormal':
        # For log-normal, we use the base_value as the mean and derive std dev from variation_scale and randomness_scale
        target_mean = base_value
        target_std = base_value * variation_scale * randomness_scale

        # Calculate log-normal parameters
        sigma = np.sqrt(np.log(1 + (target_std**2 / target_mean**2)))
        mu = np.log(target_mean**2 / np.sqrt(target_std**2 + target_mean**2))

        return np.random.lognormal(mu, sigma)
    else:
        raise ValueError("Distribution must be either 'normal' or 'lognormal'")

def calculate_adjusted_growth_rates(target_cagr, years, market_decay_rate):
    """
    Calculate annual growth rates that result in the target CAGR while following a declining pattern.

    Args:
    target_cagr (float): The desired overall CAGR.
    years (int): Number of years to calculate growth rates for.
    market_decay_rate (float): Rate at which the growth declines.

    Returns:
    numpy.array: Array of annual growth rates.
    """
    def objective(initial_rate):
        rates = initial_rate * np.exp(-market_decay_rate * np.arange(years))
        overall_growth = np.prod(1 + rates) - 1
        return (overall_growth - (1 + target_cagr)**years + 1)**2

    # Find the optimal initial rate
    result = minimize_scalar(objective)
    optimal_initial_rate = result.x

    # Calculate the growth rates
    growth_rates = optimal_initial_rate * np.exp(-market_decay_rate * np.arange(years))

    return growth_rates

def calculate_marketing_effectiveness(marketing_spend, marketing_roi, scale_growth_factor, market_share):
    """Calculate marketing effectiveness for competitors."""
    return marketing_spend * marketing_roi * (scale_growth_factor * (1 + market_share))

def update_market_size(previous_size, growth_rate, randomness_scale):
    """Update market size based on growth rate and randomness."""
    return previous_size * (1 + apply_stochastic_variation(growth_rate, 0.05, randomness_scale, 'normal'))

def adjust_retention_rate(base_rate, year, start_year, min_rate_factor, max_rate_factor, retention_decay_rate):
    """
    Adjust the retention rate based on the year using a modified exponential function.
    """
    years_passed = year - start_year
    adjustment = 1 - np.exp(-retention_decay_rate * years_passed)

    min_rate = min_rate_factor * base_rate
    max_rate = max_rate_factor * base_rate

    adjusted_rate = min_rate + (max_rate - min_rate) * adjustment

    return adjusted_rate

def update_competitor_metrics(competitors, randomness_scale, current_year, start_year, min_rate_factor, max_rate_factor, retention_decay_rate):
    """Update competitor metrics with stochastic variations."""
    for competitor in competitors:
        # Update marketing spend
        competitor['marketing_spend'] *= (1 + apply_stochastic_variation(
            competitor['spend_growth'], 0.05, randomness_scale, 'normal'))

        # Update marketing ROI using log-normal distribution
        competitor['marketing_roi'] = apply_stochastic_variation(
            competitor['marketing_roi'], 0.05, randomness_scale, 'lognormal')

        # Apply the new retention rate adjustment
        base_retention_rate = competitor['base_retention_rate']
        adjusted_retention_rate = adjust_retention_rate(base_retention_rate, current_year, start_year,
                                                       min_rate_factor, max_rate_factor, retention_decay_rate)

        # Apply stochastic variation to the adjusted retention rate
        competitor['retention_rate'] = apply_stochastic_variation(
            adjusted_retention_rate, 0.05, randomness_scale, 'normal')

        # Ensure retention_rate doesn't go below 0
        competitor['retention_rate'] = max(competitor['retention_rate'], 0)

    return competitors

def run_simulation(market_size_2025, market_cagr, market_decay_rate,
                   competitors, simulation_years, randomness_scale,
                   min_rate_factor, max_rate_factor):
    """
    Simulates the Brazilian iGaming market dynamics from 2025 to 2030.
    Uses vectorized operations where possible for improved performance.
    """
    # Calculate growth rates based on CAGR and decay rate
    growth_rates = calculate_adjusted_growth_rates(market_cagr, simulation_years, market_decay_rate)

    # Initialize market size with stochastic variation
    market_size = apply_stochastic_variation(market_size_2025 * 1e9, 0.05, randomness_scale, 'normal')

    # Pre-allocate arrays for efficiency
    years = np.arange(2025, 2025 + simulation_years)
    market_sizes = np.zeros(simulation_years)
    results = []

    # Initialize competitor metrics
    for competitor in competitors:
        competitor['revenue'] = market_size * competitor['market_share']
        # Store the initial retention rate as base_retention_rate
        competitor['base_retention_rate'] = competitor['retention_rate']

    # Main simulation loop
    for year_idx in range(simulation_years):
        current_year = years[year_idx]

        # Update market size
        market_size = update_market_size(market_size, growth_rates[year_idx], randomness_scale)
        market_sizes[year_idx] = market_size

        # Update competitor metrics
        competitors = update_competitor_metrics(competitors, randomness_scale, current_year, years[0],
                                                min_rate_factor, max_rate_factor, retention_decay_rate)

        # Calculate retained revenue
        retained_revenues = np.array([comp['revenue'] * comp['retention_rate'] for comp in competitors])

        # Calculate available market revenue
        total_available_revenue = max(market_size - np.sum(retained_revenues), 0)

        # Calculate marketing effectiveness
        marketing_effectiveness = np.array([
            calculate_marketing_effectiveness(
                comp['marketing_spend'], comp['marketing_roi'],
                comp['scale_growth_factor'], comp['market_share']
            ) for comp in competitors
        ])

        # Distribute new revenue
        total_effectiveness = np.sum(marketing_effectiveness)
        new_revenues = (marketing_effectiveness / total_effectiveness) * total_available_revenue if total_effectiveness > 0 else np.zeros_like(marketing_effectiveness)

        # Update competitor revenues and market shares
        for idx, competitor in enumerate(competitors):
            competitor['revenue'] = retained_revenues[idx] + new_revenues[idx]
            competitor['market_share'] = competitor['revenue'] / market_size

            # Store results
            results.append({
                'year': current_year,
                'competitor': competitor['name'],
                'revenue': competitor['revenue'],
                'market_share': competitor['market_share'],
                'retention_rate': competitor['retention_rate'],
                'marketing_roi': competitor['marketing_roi']
            })

    return pd.DataFrame(results)

"""# Run simulations"""

##### Chunk 3: Configure Competitor Data with Widgets & Run Simulations

# Number of simulations widget
num_simulations = widgets.IntSlider(value=1000, min = 10, max = 10000, step=100,
                                    description='N Sims:',
                                    style={'description_width': 'initial'},
                                    layout=widgets.Layout(width='300px'))

# Market Data Widgets
market_size_2025 = widgets.FloatSlider(value=5.2, min = 1.0, max = 20.0, step=0.1,
                                       description='Mkt Sz ($B):',
                                       style={'description_width': 'initial'},
                                       layout=widgets.Layout(width='300px'))

market_cagr = widgets.FloatSlider(value=0.15, min = 0.0, max = 1.0, step=0.01,
                                  description='Mkt CAGR:',
                                  style={'description_width': 'initial'},
                                  layout=widgets.Layout(width='300px'))

market_decay_rate = widgets.FloatSlider(value=0.2, min=0.0, max=1.0, step=0.01,
                                        description='CAGR Decay:',
                                        style={'description_width': 'initial'},
                                        layout=widgets.Layout(width='300px'))


# Randomness Scale Widget
randomness_scale = widgets.FloatSlider(value=3.0, min=1.0, max=5.0, step=0.1,
                                       description='Stochastic Var.',
                                       style={'description_width': 'initial'},
                                       layout=widgets.Layout(width='300px'))

print(f"Sim Setup Variables")
display(VBox([num_simulations, market_size_2025, market_cagr, market_decay_rate, randomness_scale]))


# Retention rate adjustment widgets
min_retention_factor = widgets.FloatSlider(value=0.6, min=0.1, max=0.9, step=0.05,
                                           description='Min Retention Factor:',
                                           style={'description_width': 'initial'},
                                           layout=widgets.Layout(width='300px'))

max_retention_factor = widgets.FloatSlider(value=1.1, min=1.0, max=2.0, step=0.05,
                                           description='Max Retention Factor:',
                                           style={'description_width': 'initial'},
                                           layout=widgets.Layout(width='300px'))

retention_decay_rate = widgets.FloatSlider(value=0.6, min=0.1, max=2.0, step=0.1,
                                           description='Retention Decay:',
                                           style={'description_width': 'initial'},
                                           layout=widgets.Layout(width='300px'))


print(f"Retention Rate Adjustment Parameters")
display(VBox([min_retention_factor, max_retention_factor, retention_decay_rate]))

# Competitor Market Shares Widgets
competitor_market_shares_widgets = {
    'Betano': widgets.FloatText(value=0.23, description='Betano:', step=0.01, layout=widgets.Layout(width='200px')),
    'Bet365': widgets.FloatText(value=0.20, description='Bet365:', step=0.01, layout=widgets.Layout(width='200px')),
    'BetNacional': widgets.FloatText(value=0.10, description='BetNacional:', step=0.01, layout=widgets.Layout(width='200px')),
    'Superbet': widgets.FloatText(value=0.05, description='Superbet:', step=0.01, layout=widgets.Layout(width='200px')),
    'Others': widgets.FloatText(value=0, description='Others:', step=0.01, layout=widgets.Layout(width='200px'), disabled=True)
}

def update_others_share(*args):
    others_share = 1.0 - sum(widget.value for name, widget in competitor_market_shares_widgets.items() if name != 'Others')
    competitor_market_shares_widgets['Others'].value = max(others_share, 0)

for name, widget in competitor_market_shares_widgets.items():
    if name != 'Others':
        widget.observe(update_others_share, 'value')

update_others_share()

print(f"Initial Market Share")
display(VBox([widget for widget in competitor_market_shares_widgets.values()]))

# Create widgets for each competitor's parameters
competitor_widgets = {}
for name in competitor_market_shares_widgets.keys():
    competitor_widgets[name] = {
        'marketing_spend': widgets.FloatText(value=100, step=1,
                                             description='Spend (M$):',
                                             style={'description_width': 'initial'},
                                             layout=widgets.Layout(width='200px')),

        'marketing_roi': widgets.FloatText(value=1.5, step=0.1,
                                           description='Spend ROI:',
                                           style={'description_width': 'initial'},
                                           layout=widgets.Layout(width='200px')),

        'scale_growth_factor': widgets.FloatText(value=1.0,  step=0.01,
                                                 description='Scale Factor:',
                                                 style={'description_width': 'initial'},
                                                 layout=widgets.Layout(width='200px')),

        'retention_rate': widgets.FloatText(value=0.5,
                                            description='Retention %:',
                                            step=0.01,
                                            style={'description_width': 'initial'},
                                            layout=widgets.Layout(width='200px')),

        'spend_growth': widgets.FloatText(value=0.15,
                                          description='Spend CAGR:',
                                          step=0.01,
                                          style={'description_width': 'initial'},
                                          layout=widgets.Layout(width='200px'))
    }

for name, widget_dict in competitor_widgets.items():
    print(f"{name} Parameters")
    display(HBox([widget for widget in widget_dict.values()]))

def get_competitors():
    """Gather competitor data from widgets into a list of dictionaries."""
    return [
        {
            'name': name,
            'market_share': competitor_market_shares_widgets[name].value,
            'marketing_spend': competitor_widgets[name]['marketing_spend'].value * 1e6,  # Convert to actual value
            'marketing_roi': competitor_widgets[name]['marketing_roi'].value,
            'scale_growth_factor': competitor_widgets[name]['scale_growth_factor'].value,
            'retention_rate': competitor_widgets[name]['retention_rate'].value,
            'spend_growth': competitor_widgets[name]['spend_growth'].value
        } for name in competitor_market_shares_widgets.keys()
    ]

def run_multiple_simulations():
    all_results = []
    for _ in range(num_simulations.value):
        simulation_result = run_simulation(
            market_size_2025.value,
            market_cagr.value,
            market_decay_rate.value,
            get_competitors(),
            simulation_years,
            randomness_scale.value,
            min_retention_factor.value,
            max_retention_factor.value,
            retention_decay_rate.value
        )
        all_results.append(simulation_result)

    return pd.concat(all_results, ignore_index=True)

# Button to run simulations
run_button = widgets.Button(description="Run Simulations", button_style='success')
output = widgets.Output()

def on_button_clicked(b):
    with output:
        output.clear_output()
        print("Running simulations...")
        results_df = run_multiple_simulations()
        print("Simulations completed. Generating summary...")

        # Summary Statistics
        summary = results_df.groupby(['year', 'competitor']).agg({
            'market_share': ['mean', 'std']
        }).reset_index()
        summary.columns = ['year', 'competitor', 'market_share_mean', 'market_share_std']

        # Plotting
        plt.figure(figsize=(8, 6))
        sns.set_style("whitegrid")

        initial_year = 2024
        for competitor_name in competitor_market_shares_widgets.keys():
            competitor_data = summary[summary['competitor'] == competitor_name]
            years = [initial_year] + list(competitor_data['year'])
            market_shares = [competitor_market_shares_widgets[competitor_name].value] + list(competitor_data['market_share_mean'])
            plt.plot(years, market_shares, label=competitor_name, marker='o')
            std_dev = competitor_data['market_share_std'].fillna(0)
            plt.fill_between(years[1:],
                             [market_shares[1:][i] - 1*std_dev.iloc[i] for i in range(len(std_dev))],
                             [market_shares[1:][i] + 1*std_dev.iloc[i] for i in range(len(std_dev))],
                             alpha=0.2)

        plt.xlabel('Year')
        plt.ylabel('Average Market Share')
        plt.title('Average Market Share Over Time (2024-2030)')
        plt.legend()
        plt.show()

run_button.on_click(on_button_clicked)
display(run_button, output)

"""# Sensitivity Analysis"""

##### Chunk 4: Sensitivity Analysis

def run_multi_parameter_sensitivity_analysis(parameters, base_values, variation, steps):
    variation_factor = variation / 100
    all_results = []
    years = range(2025, 2031)  # Simulation years from 2025 to 2030

    for parameter in parameters:
        base_value = base_values[parameter]
        param_values = np.linspace(base_value * (1 - variation_factor),
                                   base_value * (1 + variation_factor),
                                   steps)

        for param_value in param_values:
            if parameter in competitor_widgets['Superbet']:
                competitor_widgets['Superbet'][parameter].value = param_value
            elif parameter == 'min_retention_factor':
                min_retention_factor.value = param_value
            elif parameter == 'max_retention_factor':
                max_retention_factor.value = param_value
            elif parameter == 'retention_decay_rate':
                retention_decay_rate.value = param_value
            elif parameter == 'market_decay_rate':
                market_decay_rate.value = param_value

            # Run simulations for all years
            for year in years:
                sim_results = run_multiple_simulations()
                superbet_data = sim_results[sim_results['competitor'] == 'Superbet']
                year_data = superbet_data[superbet_data['year'] == year]

                result = {
                    'parameter': parameter,
                    'parameter_value': param_value,
                    'year': year,
                    'market_share_mean': year_data['market_share'].mean(),
                    'market_share_std': year_data['market_share'].std()
                }
                all_results.append(result)

    # Reset parameters to original values
    for param in parameters:
        if param in competitor_widgets['Superbet']:
            competitor_widgets['Superbet'][param].value = base_values[param]
        elif param == 'min_retention_factor':
            min_retention_factor.value = base_values[param]
        elif param == 'max_retention_factor':
            max_retention_factor.value = base_values[param]
        elif parameter == 'retention_decay_rate':
            retention_decay_rate.value = param_value
        elif parameter == 'market_decay_rate':
            market_decay_rate.value = param_value

    return pd.DataFrame(all_results)


# Create the sensitivity table
def create_sensitivity_heatmap(sensitivity_df, param1, param2):
    """Create a compact heatmap for two parameters with rounded x-axis labels."""
    # Filter data for the two selected parameters and the last year (2030)
    last_year = 2030
    filtered_data = sensitivity_df[(sensitivity_df['year'] == last_year) &
                                   (sensitivity_df['parameter'].isin([param1, param2]))]

    if filtered_data.empty:
        print("Error: No data available for the selected parameters in the year 2030.")
        return None

    # Separate data for each parameter
    param1_data = filtered_data[filtered_data['parameter'] == param1]
    param2_data = filtered_data[filtered_data['parameter'] == param2]

    # Get unique parameter values
    param1_values = param1_data['parameter_value'].unique()
    param2_values = param2_data['parameter_value'].unique()

    # Create a new dataframe with all combinations of parameter values
    heatmap_data = []
    for val1 in param1_values:
        for val2 in param2_values:
            market_share1 = param1_data[param1_data['parameter_value'] == val1]['market_share_mean'].values[0]
            market_share2 = param2_data[param2_data['parameter_value'] == val2]['market_share_mean'].values[0]
            heatmap_data.append({
                param1: val1,
                param2: val2,
                'market_share': (market_share1 + market_share2) / 2
            })

    heatmap_df = pd.DataFrame(heatmap_data)

    # Create pivot table for heatmap
    pivot_table = heatmap_df.pivot(index=param1, columns=param2, values='market_share')

    # Round values to 3 decimal places (for percentage display)
    pivot_table = pivot_table.round(3)

    # Create a custom colormap
    colors = ['#FFA500', '#FFFFFF', '#0000FF']  # Orange, White, Blue
    n_bins = 100  # Number of color gradations
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

    # Create the heatmap
    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(pivot_table, annot=True, fmt='.1%', cmap=cmap,
                     center=0.2,  # Center color at 20%
                     vmin=0.0, vmax=0.6,
                     cbar_kws={'label': 'Market Share'}, linewidths=0.5,
                     annot_kws={'size': 8})

    # Set labels and title
    plt.title(f"Market Share Sensitivity", fontsize=10)
    plt.xlabel(param2, fontsize=8)
    plt.ylabel(param1, fontsize=8)

    # Set x-axis and y-axis labels
    rounded_param2_values = [f"{val:.2f}" for val in param2_values]  # Round to 2 decimal places
    ax.set_xticklabels(rounded_param2_values, rotation=45, ha='right', fontsize=6)
    ax.set_yticklabels(param1_values, rotation=0, fontsize=6)

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    return plt


def visualize_multi_parameter_sensitivity(sensitivity_df, initial_market_share):
    """
    Visualize multi-parameter sensitivity including the initial 2024 market share.

    :param sensitivity_df: DataFrame containing sensitivity analysis results
    :param initial_market_share: Initial market share for Superbet in 2024
    :return: List of (figure, axes) tuples
    """
    parameters = sensitivity_df['parameter'].unique()
    years = sensitivity_df['year'].unique()

    # Create a single figure with subplots
    fig, axes = plt.subplots(1, len(parameters), figsize=(6*len(parameters), 5), squeeze=False)
    axes = axes.flatten()  # Flatten axes array for easier indexing

    for idx, parameter in enumerate(parameters):
        ax = axes[idx]
        sns.set_style("whitegrid")

        param_data = sensitivity_df[sensitivity_df['parameter'] == parameter]
        param_values = param_data['parameter_value'].unique()

        for param_value in param_values:
            value_data = param_data[param_data['parameter_value'] == param_value]

            # Add the initial 2024 market share
            plot_years = [2024] + list(value_data['year'])
            plot_market_shares = [initial_market_share] + list(value_data['market_share_mean'])

            ax.plot(plot_years, plot_market_shares,
                    label=f"{param_value:.2f}", marker='o', markersize=4)

            # Add shaded area for standard deviation (excluding 2024)
            ax.fill_between(value_data['year'],
                            value_data['market_share_mean'] - value_data['market_share_std'],
                            value_data['market_share_mean'] + value_data['market_share_std'],
                            alpha=0.1)

        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Superbet Market Share' if idx == 0 else '', fontsize=8)
        ax.set_title(f'Impact of {parameter}', fontsize=10)

        # Set x-axis ticks to show all years
        ax.set_xticks(plot_years)
        ax.set_xticklabels(plot_years, rotation=45, ha='right', fontsize=8)

        # Format y-axis as percentages
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

        # Adjust legend
        ax.legend(title=parameter, loc='best', fontsize=8, title_fontsize=9,
                  bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)

        # Adjust tick label size
        ax.tick_params(axis='both', which='major', labelsize=8)

    plt.tight_layout()
    return [(fig, axes)]


def calculate_impact_summary(sensitivity_df):
    parameters = sensitivity_df['parameter'].unique()
    impact_summary = []

    for parameter in parameters:
        param_data = sensitivity_df[sensitivity_df['parameter'] == parameter]
        min_value = param_data['parameter_value'].min()
        max_value = param_data['parameter_value'].max()

        # Get the market share for the last year (2030) for each parameter value
        last_year_data = param_data[param_data['year'] == param_data['year'].max()]
        min_share = last_year_data[last_year_data['parameter_value'] == min_value]['market_share_mean'].values[0]
        max_share = last_year_data[last_year_data['parameter_value'] == max_value]['market_share_mean'].values[0]

        impact_range = max_share - min_share

        impact_summary.append({
            'Parameter': parameter,
            'Min Value': min_value,
            'Max Value': max_value,
            'Min Share': min_share,
            'Max Share': max_share,
            'Impact Range': abs(impact_range)  # Use absolute value for consistent ordering
        })

    return pd.DataFrame(impact_summary).sort_values('Impact Range', ascending=False)


# Display all widgets from Chunk 3 for user adjustment
print("Adjust simulation parameters if needed:")
display(VBox([
    num_simulations,
    market_size_2025,
    market_cagr,
    market_decay_rate,
    randomness_scale
]))

print("Retention Rate Adjustment Parameters:")
display(VBox([min_retention_factor, max_retention_factor, retention_decay_rate]))

print("Competitor Market Shares:")
display(VBox([widget for widget in competitor_market_shares_widgets.values()]))

for name, widget_dict in competitor_widgets.items():
    print(f"{name} Parameters:")
    display(HBox([widget for widget in widget_dict.values()]))

# Sensitivity Analysis Widgets
print("Sensitivity Analysis Configuration:")
parameters_to_analyze = widgets.SelectMultiple(
    options=['marketing_spend', 'marketing_roi', 'scale_growth_factor', 'retention_rate', 'spend_growth',
             'min_retention_factor', 'max_retention_factor', 'retention_decay_rate', 'market_decay_rate'],
    value=['marketing_spend'],
    description='Parameters:',
    layout=widgets.Layout(width='300px')
)
print("Select 1-3 parameters for sensitivity analysis:")
display(parameters_to_analyze)

variation_percentage = widgets.FloatSlider(value=20, min=1, max=50, step=1,
                                           description='Variation (%):',
                                           layout=widgets.Layout(width='300px'))

num_steps = widgets.IntSlider(value=5, min=3, max=10, step=1,
                              description='Steps:',
                              layout=widgets.Layout(width='300px'))

sensitivity_widgets = widgets.VBox([
    variation_percentage,
    num_steps
])

display(sensitivity_widgets)

# Button to run sensitivity analysis
run_sensitivity_button = widgets.Button(description="Run Sensitivity Analysis",
                                        button_style='success',
                                        layout=widgets.Layout(width='300px'))

sensitivity_output = widgets.Output()

def on_sensitivity_button_clicked(b):
    with sensitivity_output:
        clear_output(wait=True)

        selected_parameters = parameters_to_analyze.value
        if len(selected_parameters) < 1 or len(selected_parameters) > 3:
            print("Please select 1 to 3 parameters for the sensitivity analysis.")
            return

        print("Running multi-parameter sensitivity analysis...")

        try:
            base_values = {
                param: competitor_widgets['Superbet'][param].value
                if param in competitor_widgets['Superbet']
                else globals()[param].value
                for param in selected_parameters
            }
            sensitivity_df = run_multi_parameter_sensitivity_analysis(
                selected_parameters,
                base_values,
                variation_percentage.value,
                num_steps.value
            )

            print("Sensitivity analysis completed. Generating visualizations...")

            print("\nImpact Summary:")
            impact_summary = calculate_impact_summary(sensitivity_df)
            display(HTML(impact_summary.to_html(index=False)))

            print("\nParameter Sensitivity Visualization:")
            initial_market_share = competitor_market_shares_widgets['Superbet'].value
            plots = visualize_multi_parameter_sensitivity(sensitivity_df, initial_market_share)
            for fig, _ in plots:
                plt.figure(fig.number)
                plt.show()

            if len(selected_parameters) == 2:
                print("\nSensitivity Heatmap:")
                heatmap = create_sensitivity_heatmap(sensitivity_df, selected_parameters[0], selected_parameters[1])
                if heatmap:
                    plt.show()
                else:
                    print("Failed to create heatmap. See debug output above.")

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            raise

run_sensitivity_button.on_click(on_sensitivity_button_clicked)

display(run_sensitivity_button, sensitivity_output)