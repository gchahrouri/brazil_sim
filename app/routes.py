from flask import render_template, request, jsonify
from app import app
from app.simulation import run_simulation, run_sensitivity_analysis
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import plotly.figure_factory as ff
import logging

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_simulation', methods=['POST'])
def run_simulation_route():
    params = {
        'market_size_2025': float(request.form['market_size_2025']),
        'market_cagr': float(request.form['market_cagr']),
        'market_decay_rate': float(request.form['market_decay_rate']),
        'randomness_scale': float(request.form['randomness_scale']),
        'num_simulations': int(request.form['num_simulations']),
        'min_retention_factor': float(request.form['min_retention_factor']),
        'max_retention_factor': float(request.form['max_retention_factor']),
        'retention_decay_rate': float(request.form['retention_decay_rate']),
    }

    competitors = []
    named_companies = ['Betano', 'Bet365', 'BetNacional', 'Superbet']
    total_named_share = 0
    for company in named_companies:
        market_share = float(request.form[f'{company}_market_share'])
        total_named_share += market_share
        competitors.append({
            'name': company,
            'market_share': market_share,
            'marketing_spend': float(request.form[f'{company}_marketing_spend']) * 1e6,
            'marketing_roi': float(request.form[f'{company}_marketing_roi']),
            'scale_growth_factor': float(request.form[f'{company}_scale_growth_factor']),
            'retention_rate': float(request.form[f'{company}_retention_rate']),
            'spend_growth': float(request.form[f'{company}_spend_growth']),
        })
    
    # Calculate and add "Others" market share
    others_market_share = max(0, 1 - total_named_share)
    competitors.append({
        'name': 'Others',
        'market_share': others_market_share,
        'marketing_spend': float(request.form['Others_marketing_spend']) * 1e6,
        'marketing_roi': float(request.form['Others_marketing_roi']),
        'scale_growth_factor': float(request.form['Others_scale_growth_factor']),
        'retention_rate': float(request.form['Others_retention_rate']),
        'spend_growth': float(request.form['Others_spend_growth']),
    })

    results = run_simulation(
        params['market_size_2025'], params['market_cagr'], params['market_decay_rate'],
        competitors, 6, params['randomness_scale'],
        params['min_retention_factor'], params['max_retention_factor'],
        params['retention_decay_rate'], params['num_simulations']
    )

    logging.debug(f"Simulation results shape: {results.shape}")
    logging.debug(f"Simulation results head: {results.head()}")
    
    # Debug: Print summary statistics for raw results
    logging.debug("Raw simulation results summary:")
    logging.debug(results.groupby(['year', 'competitor'])['market_share'].describe())
    
    # Debug: Check if market shares sum to 1 for each year and simulation
    market_share_sums = results.groupby(['simulation', 'year'])['market_share'].sum()
    logging.debug("Market share sums:")
    logging.debug(market_share_sums.describe())
    logging.debug(f"Min sum: {market_share_sums.min()}, Max sum: {market_share_sums.max()}")

    # Correct aggregation of market shares
    summary = results.groupby(['year', 'competitor'])['market_share'].agg(['mean', 'std']).reset_index()
    summary.columns = ['year', 'competitor', 'market_share_mean', 'market_share_std']

    logging.debug(f"Summary shape: {summary.shape}")
    logging.debug(f"Summary head: {summary.head()}")
    
    # Debug: Check if aggregated market shares sum to 1 for each year
    agg_market_share_sums = summary.groupby('year')['market_share_mean'].sum()
    logging.debug("Aggregated market share sums:")
    logging.debug(agg_market_share_sums)

    # Add initial market shares for 2024
    initial_data = [{'year': 2024, 'competitor': comp['name'], 'market_share_mean': comp['market_share'], 'market_share_std': 0} for comp in competitors]
    summary = pd.concat([pd.DataFrame(initial_data), summary], ignore_index=True)

    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)', 'rgb(148, 103, 189)']
    
    fig = go.Figure()
    
    for i, competitor in enumerate(summary['competitor'].unique()):
        competitor_data = summary[summary['competitor'] == competitor]
        color = colors[i % len(colors)]
        
        # Main line (including 2024)
        fig.add_trace(go.Scatter(
            x=competitor_data['year'],
            y=competitor_data['market_share_mean'],
            mode='lines+markers',
            name=competitor,
            line=dict(color=color),
            marker=dict(size=8, color=color),
            hovertemplate='Year: %{x}<br>Company: ' + competitor + '<br>Market Share: %{y:.2%}<br>Std Dev: %{text:.2%}<extra></extra>',
            text=competitor_data['market_share_std']
        ))
        
        # Shaded area for standard deviation (starting from 2025)
        fig.add_trace(go.Scatter(
            x=competitor_data['year'],
            y=competitor_data['market_share_mean'] + competitor_data['market_share_std'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=competitor_data['year'],
            y=competitor_data['market_share_mean'] - competitor_data['market_share_std'],
            mode='lines',
            line=dict(width=0),
            fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.2)'),
            fill='tonexty',
            showlegend=False,
            hoverinfo='skip'
        ))

    max_market_share = summary['market_share_mean'].max()
    y_max = min(1, max_market_share * 1.2)  # Set y-axis max to 120% of the highest value, but not exceeding 1

    fig.update_layout(
        title='Simulated Market Share (2024-2030)',  
        xaxis_title='Year',
        yaxis_title='Market Share',
        yaxis_tickformat='.1%',
        yaxis_range=[0, y_max],
        xaxis=dict(tickmode='array', tickvals=list(range(2024, 2031))),
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.05), 
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=600, 
        margin=dict(r=150)  
    )

    plot_div = pio.to_html(fig, full_html=False)
    return plot_div

@app.route('/run_sensitivity', methods=['POST'])
def run_sensitivity_route():
    params = {
        'market_size_2025': float(request.form['market_size_2025']),
        'market_cagr': float(request.form['market_cagr']),
        'market_decay_rate': float(request.form['market_decay_rate']),
        'randomness_scale': float(request.form['randomness_scale']),
        'num_simulations': int(request.form['num_simulations']),
        'min_retention_factor': float(request.form['min_retention_factor']),
        'max_retention_factor': float(request.form['max_retention_factor']),
        'retention_decay_rate': float(request.form['retention_decay_rate']),
    }

    competitors = []
    for company in ['Betano', 'Bet365', 'BetNacional', 'Superbet', 'Others']:
        competitors.append({
            'name': company,
            'market_share': float(request.form[f'{company}_market_share']),
            'marketing_spend': float(request.form[f'{company}_marketing_spend']) * 1e6,
            'marketing_roi': float(request.form[f'{company}_marketing_roi']),
            'scale_growth_factor': float(request.form[f'{company}_scale_growth_factor']),
            'retention_rate': float(request.form[f'{company}_retention_rate']),
            'spend_growth': float(request.form[f'{company}_spend_growth']),
        })

    sensitivity_params = request.form.getlist('sensitivity_params')
    if len(sensitivity_params) < 1 or len(sensitivity_params) > 3:
        return "Please select 1 to 3 parameters for sensitivity analysis."

    # Superbet-specific parameters for sensitivity analysis
    superbet_params = {
        'marketing_spend': competitors[3]['marketing_spend'],
        'marketing_roi': competitors[3]['marketing_roi'],
        'scale_growth_factor': competitors[3]['scale_growth_factor'],
        'retention_rate': competitors[3]['retention_rate'],
        'spend_growth': competitors[3]['spend_growth']
    }

    sensitivity_steps = int(request.form['sensitivity_steps'])
    sensitivity_variation = float(request.form['sensitivity_variation']) / 100  # Convert to decimal

    sensitivity_results = run_sensitivity_analysis(params, superbet_params, sensitivity_params, competitors, sensitivity_steps, sensitivity_variation)

    figs = []

    # Line graph for each parameter
    if len(sensitivity_params) == 1:
        param = sensitivity_params[0]
        fig = go.Figure()
        
        for year in sensitivity_results['year'].unique():
            year_data = sensitivity_results[sensitivity_results['year'] == year]
            fig.add_trace(go.Scatter(
                x=year_data[f'{param}_value'],
                y=year_data['market_share_mean'],
                mode='lines+markers',
                name=f'Year {year}',
                error_y=dict(
                    type='data',
                    array=year_data['market_share_std'],
                    visible=True
                )
            ))

        fig.update_layout(
            title=f'Sensitivity Analysis: Superbet {param}',
            xaxis_title=param,
            yaxis_title='Market Share',
            yaxis_tickformat='.1%',
            yaxis_range=[0, 1],
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.05),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=600,
            margin=dict(r=150)
        )
        figs.append(pio.to_html(fig, full_html=False))

    # Heatmap (only when 2 parameters are selected)
    if len(sensitivity_params) == 2:
        param1, param2 = sensitivity_params
        heatmap_data = sensitivity_results[sensitivity_results['year'] == 2030].copy()
        
        # Create pivot table
        pivot_table = heatmap_data.pivot_table(
            values='market_share_mean', 
            index=f'{param1}_value', 
            columns=f'{param2}_value', 
            aggfunc='mean'
        )
        
        # Round market share values to 1 decimal place
        pivot_table = pivot_table.round(3) * 100  # Convert to percentage and round to 1 decimal place
        
        # Create custom colorscale
        colorscale = [
            [0, 'rgb(255, 127, 14)'],  # Orange for values below 20%
            [0.333, 'rgb(255, 255, 255)'],  # White for 20%
            [1, 'rgb(31, 119, 180)']  # Blue for values above 20%, deepest at 60%
        ]
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale=colorscale,
            zmin=0,
            zmax=60,
            colorbar=dict(title='Market Share (%)'),
            text=[[f'{val:.1f}%' for val in row] for row in pivot_table.values],
            hovertemplate='%{y}, %{x}<br>Market Share: %{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Sensitivity Heatmap: Superbet Market Share in 2030',
            xaxis_title=f'{param2}',
            yaxis_title=f'{param1}',
            width=800,
            height=600,
            yaxis=dict(autorange='reversed')  # This will flip the y-axis
        )
        
        # Add text annotations
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                fig.add_annotation(
                    x=pivot_table.columns[j],
                    y=pivot_table.index[i],
                    text=f'{pivot_table.values[i, j]:.1f}%',
                    showarrow=False,
                    font=dict(color='black', size=10)
                )
        
        figs.append(pio.to_html(fig, full_html=False))
    
    return ''.join(figs)
