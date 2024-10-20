from flask import render_template, request, jsonify, make_response
from app import app
from app.simulation import run_simulation, run_sensitivity_analysis
from app.tracking import log_action, get_unique_users, get_action_count, get_parameter_changes, get_analytics_data
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import logging
from plotly.subplots import make_subplots

@app.route('/')
def index():
    response = make_response(render_template('index.html'))
    if 'user_id' not in request.cookies:
        user_id = log_action('page_view')
        response.set_cookie('user_id', user_id)
    return response

@app.route('/run_simulation', methods=['POST'])
def run_simulation_route():
    log_action('run_simulation')
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

    summary = results.groupby(['year', 'competitor'])['market_share'].agg(['mean', 'std']).reset_index()
    summary.columns = ['year', 'competitor', 'market_share_mean', 'market_share_std']

    initial_data = [{'year': 2024, 'competitor': comp['name'], 'market_share_mean': comp['market_share'], 'market_share_std': 0} for comp in competitors]
    summary = pd.concat([pd.DataFrame(initial_data), summary], ignore_index=True)

    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)', 'rgb(148, 103, 189)']
    
    fig = go.Figure()
    
    for i, competitor in enumerate(summary['competitor'].unique()):
        competitor_data = summary[summary['competitor'] == competitor]
        color = colors[i % len(colors)]
        
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
        
        fig.add_trace(go.Scatter(
            x=competitor_data['year'].tolist() + competitor_data['year'].tolist()[::-1],
            y=(competitor_data['market_share_mean'] + competitor_data['market_share_std']).tolist() + 
               (competitor_data['market_share_mean'] - competitor_data['market_share_std']).tolist()[::-1],
            fill='toself',
            fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.2)'),
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Add annotation at the end of each line
        fig.add_annotation(
            x=competitor_data['year'].iloc[-1],
            y=competitor_data['market_share_mean'].iloc[-1],
            text=f"{competitor}: {competitor_data['market_share_mean'].iloc[-1]:.1%}",
            showarrow=False,
            xanchor='left',
            xshift=10,
            font=dict(color=color)
        )

    max_market_share = summary['market_share_mean'].max()
    y_max = min(1, max_market_share * 1.2)

    fig.update_layout(
        title='Simulated Market Share (2024-2030)',  
        xaxis_title='Year',
        yaxis_title='Market Share',
        yaxis_tickformat='.1%',
        yaxis_range=[0, y_max],
        xaxis=dict(tickmode='array', tickvals=list(range(2024, 2031))),
        showlegend=False,  # Hide the legend since we're using annotations
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=600, 
        margin=dict(r=150)  
    )

    plot_div = pio.to_html(fig, full_html=False)
    return plot_div

@app.route('/run_sensitivity', methods=['POST'])
def run_sensitivity_route():
    log_action('run_sensitivity')
    
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

    superbet_params = {
        'marketing_spend': competitors[3]['marketing_spend'],
        'marketing_roi': competitors[3]['marketing_roi'],
        'scale_growth_factor': competitors[3]['scale_growth_factor'],
        'retention_rate': competitors[3]['retention_rate'],
        'spend_growth': competitors[3]['spend_growth']
    }

    logging.info(f"Sensitivity parameters: {sensitivity_params}")
    logging.info(f"Superbet parameters: {superbet_params}")

    sensitivity_steps = int(request.form['sensitivity_steps'])
    sensitivity_variation = float(request.form['sensitivity_variation']) / 100

    sensitivity_results = run_sensitivity_analysis(params, superbet_params, sensitivity_params, competitors, sensitivity_steps, sensitivity_variation)
    logging.info(f"Sensitivity results shape: {sensitivity_results.shape}")
    logging.info(f"Sensitivity results columns: {sensitivity_results.columns}")
    logging.info(f"Unique values for each parameter: {sensitivity_results[sensitivity_params].nunique().to_dict()}")

    figs = []

    # Heatmap (only when 2 parameters are selected)
    if len(sensitivity_params) == 2:
        param1, param2 = sensitivity_params
        heatmap_data = sensitivity_results[sensitivity_results['year'] == 2030].copy()
        
        param1_values = sorted(heatmap_data[param1].unique())
        param2_values = sorted(heatmap_data[param2].unique())
        
        pivot_table = heatmap_data.pivot_table(
            values='market_share_mean', 
            index=param1, 
            columns=param2, 
            aggfunc='mean'
        )
        
        pivot_table = pivot_table.round(3) * 100
        
        colorscale = [
            [0, 'rgb(255, 127, 14)'],
            [0.333, 'rgb(255, 255, 255)'],
            [1, 'rgb(31, 119, 180)']
        ]
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=param2_values,
            y=param1_values,
            colorscale=colorscale,
            zmin=0,
            zmax=50,
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
            yaxis=dict(autorange='reversed', tickvals=param1_values),
            xaxis=dict(tickvals=param2_values)
        )
        
        for i, y in enumerate(param1_values):
            for j, x in enumerate(param2_values):
                fig.add_annotation(
                    x=x,
                    y=y,
                    text=f'{pivot_table.values[i, j]:.1f}%',
                    showarrow=False,
                    font=dict(color='black', size=10)
                )
        
        figs.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))
        logging.info("Heatmap generated")

    # Create a single row for all line graphs
    fig = make_subplots(rows=1, cols=len(sensitivity_params), 
                        subplot_titles=[f'{param} Sensitivity' for param in sensitivity_params],
                        shared_yaxes=True)

    for idx, param in enumerate(sensitivity_params, start=1):
        # Get unique values for the current parameter
        param_values = sorted(sensitivity_results[param].unique())
        logging.info(f"Unique values for {param}: {param_values}")

        colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)']
        
        for i, value in enumerate(param_values):
            # Filter data for the current parameter value
            value_data = sensitivity_results[sensitivity_results[param] == value]
            
            # If we have two parameters, we need to average over the other parameter
            if len(sensitivity_params) == 2:
                other_param = [p for p in sensitivity_params if p != param][0]
                value_data = value_data.groupby('year').agg({
                    'market_share_mean': 'mean',
                    'market_share_std': 'mean'
                }).reset_index()

            logging.info(f"Data shape for {param}={value}: {value_data.shape}")

            # Add initial 2024 data point
            initial_market_share = competitors[3]['market_share']
            value_data = pd.concat([
                pd.DataFrame({'year': [2024], 'market_share_mean': [initial_market_share], 'market_share_std': [0]}),
                value_data
            ]).reset_index(drop=True)

            color = colors[i % len(colors)]

            # Format the value for display
            display_value = f"${value/1e6:.0f}M" if param == 'marketing_spend' else f"{value:.2f}"

            fig.add_trace(go.Scatter(
                x=value_data['year'],
                y=value_data['market_share_mean'],
                mode='lines+markers',
                name=f'{param}={display_value}',
                line=dict(color=color, width=2),
                marker=dict(size=8, color=color),
                showlegend=False
            ), row=1, col=idx)

            fig.add_trace(go.Scatter(
                x=value_data['year'].tolist() + value_data['year'].tolist()[::-1],
                y=(value_data['market_share_mean'] + value_data['market_share_std']).tolist() + 
                   (value_data['market_share_mean'] - value_data['market_share_std']).tolist()[::-1],
                fill='toself',
                fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.2)'),
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=idx)

            # Add annotation at the end of each line
            fig.add_annotation(
                x=value_data['year'].iloc[-1],
                y=value_data['market_share_mean'].iloc[-1],
                text=display_value,
                showarrow=False,
                xanchor='left',
                xshift=10,
                font=dict(color=color),
                row=1, col=idx
            )

        fig.update_xaxes(title_text='Year', tickmode='array', tickvals=list(range(2024, 2031)), row=1, col=idx)
        if idx == 1:
            fig.update_yaxes(title_text='Market Share', tickformat='.1%', row=1, col=idx)
        else:
            fig.update_yaxes(tickformat='.1%', row=1, col=idx)

    # Update layout
    fig.update_layout(
        title='Superbet Market Share Over Time: Sensitivity Analysis',
        height=600,
        showlegend=False,
        hovermode="x unified",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    # Adjust subplot titles
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=12)

    figs.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))
    logging.info(f"Sensitivity graphs generated for parameters: {sensitivity_params}")

    logging.info(f"Total number of graphs generated: {len(figs)}")
    return '<div class="sensitivity-graphs">' + ''.join(figs) + '</div>'

@app.route('/update_parameter', methods=['POST'])
def update_parameter():
    parameter = request.form.get('parameter')
    value = request.form.get('value')
    log_action('parameter_change', f"{parameter}:{value}")
    return jsonify(success=True)

@app.route('/analytics')
def analytics():
    analytics_data = get_analytics_data()
    parameter_changes = get_parameter_changes()

    logging.info(f"Analytics Data: {analytics_data}")
    logging.info(f"Parameter Changes: {parameter_changes}")

    return render_template('analytics.html', 
                           analytics_data=analytics_data,
                           parameter_changes=parameter_changes)

