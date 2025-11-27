"""
California Housing Price Prediction Dashboard
Option C: Simple Predictive Model + Visual Explanation

This standalone Panel app demonstrates:
1. Feature importance visualization
2. Model performance (actual vs predicted)
3. Residual analysis
4. Interactive prediction interface
5. Trust & limitations explanation
"""

import panel as pn
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from housing_predictor import HousingPredictor

# Enable Panel extensions
pn.extension('plotly', sizing_mode='stretch_width')

# Train model
print("Training model...")
predictor = HousingPredictor()
metrics = predictor.train()
print(f"✅ Model trained! R² = {metrics['r2']:.3f}, RMSE = ${metrics['rmse']*100000:,.0f}")

# Load data for visualizations
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
df = housing.frame
df.columns = [
    'median_income', 'housing_median_age', 'avg_rooms',
    'avg_bedrooms', 'population', 'avg_occupancy',
    'latitude', 'longitude', 'median_house_value'
]

# Generate predictions for visualization
X = df[['median_income', 'latitude', 'longitude']].values
y_true = df['median_house_value'].values
y_pred = predictor.model.predict(X)
residuals = y_true - y_pred

# ==============================================================================
# VISUALIZATION 1: Feature Importance
# ==============================================================================

def create_feature_importance():
    """Bar chart showing coefficient impacts."""
    coefs = metrics['coefficients']

    features = ['Income\n(per $10k)', 'Latitude\n(per degree)', 'Longitude\n(per degree)']
    impacts = [
        coefs['median_income'] * 100000,
        coefs['latitude'] * 100000,
        coefs['longitude'] * 100000
    ]
    colors = ['green' if x > 0 else 'red' for x in impacts]

    fig = go.Figure(data=[
        go.Bar(
            x=features,
            y=impacts,
            marker_color=colors,
            text=[f'${x:,.0f}' for x in impacts],
            textposition='outside'
        )
    ])

    fig.update_layout(
        title='Feature Importance: Impact on House Value',
        yaxis_title='Price Impact ($)',
        xaxis_title='Feature',
        height=400,
        showlegend=False,
        hovermode='x'
    )

    fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)

    return fig

# ==============================================================================
# VISUALIZATION 2: Actual vs Predicted
# ==============================================================================

def create_actual_vs_predicted():
    """Scatter plot showing prediction accuracy."""
    # Sample 5000 points for faster rendering
    sample_idx = np.random.choice(len(y_true), size=5000, replace=False)

    # Calculate absolute errors for color coding
    abs_errors = np.abs(residuals[sample_idx] * 100000)

    fig = go.Figure()

    # Add scatter points
    fig.add_trace(go.Scatter(
        x=y_true[sample_idx],
        y=y_pred[sample_idx],
        mode='markers',
        marker=dict(
            size=4,
            color=abs_errors,
            colorscale='YlOrRd',
            showscale=True,
            colorbar=dict(title='Error ($)'),
            opacity=0.6
        ),
        text=[f'True: ${t*100000:,.0f}<br>Pred: ${p*100000:,.0f}<br>Error: ${e:,.0f}'
              for t, p, e in zip(y_true[sample_idx], y_pred[sample_idx], abs_errors)],
        hovertemplate='%{text}<extra></extra>',
        name='Predictions'
    ))

    # Add perfect prediction line
    min_val, max_val = y_true.min(), y_true.max()
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='black', dash='dash', width=2),
        name='Perfect Prediction',
        hoverinfo='skip'
    ))

    fig.update_layout(
        title=f'Actual vs Predicted House Values (R² = {metrics["r2"]:.3f})',
        xaxis_title='Actual Value ($100k)',
        yaxis_title='Predicted Value ($100k)',
        height=500,
        hovermode='closest'
    )

    return fig

# ==============================================================================
# VISUALIZATION 3: Residual Analysis
# ==============================================================================

def create_residual_plot():
    """Residual plot showing where model struggles."""
    sample_idx = np.random.choice(len(y_pred), size=5000, replace=False)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_pred[sample_idx],
        y=residuals[sample_idx] * 100000,
        mode='markers',
        marker=dict(
            size=4,
            color=np.abs(residuals[sample_idx] * 100000),
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title='|Error| ($)'),
            opacity=0.5
        ),
        text=[f'Predicted: ${p*100000:,.0f}<br>Error: ${r*100000:,.0f}'
              for p, r in zip(y_pred[sample_idx], residuals[sample_idx])],
        hovertemplate='%{text}<extra></extra>',
        name='Residuals'
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash='dash', line_color='red', opacity=0.7,
                  annotation_text='Zero Error', annotation_position='right')

    # Add RMSE bands
    rmse_dollars = metrics['rmse'] * 100000
    fig.add_hrect(y0=-rmse_dollars, y1=rmse_dollars,
                  fillcolor='green', opacity=0.1,
                  annotation_text=f'±RMSE (${rmse_dollars:,.0f})',
                  annotation_position='top left')

    fig.update_layout(
        title='Residual Plot: Prediction Errors vs Predicted Values',
        xaxis_title='Predicted Value ($100k)',
        yaxis_title='Residual Error ($)',
        height=450,
        hovermode='closest'
    )

    return fig

# ==============================================================================
# VISUALIZATION 4: Interactive Prediction Interface
# ==============================================================================

# Region coordinates (representative points)
region_coords = {
    'San Francisco Bay Area': (37.8, -122.4),
    'Los Angeles': (34.0, -118.2),
    'San Diego': (32.7, -117.2),
    'Sacramento': (38.6, -121.5),
    'Fresno (Central Valley)': (36.7, -119.8),
    'Inland Empire': (34.0, -117.0)
}

# Prediction widgets
income_slider = pn.widgets.FloatSlider(
    name='Annual Household Income',
    start=20000,
    end=150000,
    step=5000,
    value=60000,
    format='$,.0f'
)

region_selector = pn.widgets.Select(
    name='Select Region',
    options=list(region_coords.keys()),
    value='San Francisco Bay Area'
)

proximity_toggle = pn.widgets.RadioButtonGroup(
    name='Coastal Proximity',
    options=['Coastal', 'Inland'],
    value='Coastal',
    button_type='primary'
)

@pn.depends(income_slider, region_selector, proximity_toggle)
def make_interactive_prediction(income_dollars, region, proximity):
    """Generate prediction based on user inputs."""
    # Convert income to $10k units
    income_10k = income_dollars / 10000

    # Get coordinates
    lat, lon = region_coords[region]

    # Adjust longitude for proximity
    if proximity == 'Inland':
        lon += 2.0  # Move eastward

    # Make prediction
    result = predictor.predict(income_10k, lat, lon)

    pred_value = result['predicted_value_dollars']
    conf_lower = result['confidence_lower']
    conf_upper = result['confidence_upper']

    # Compare to dataset average
    avg_value = df['median_house_value'].mean() * 100000
    diff = pred_value - avg_value
    diff_pct = (diff / avg_value) * 100
    comparison = "above" if diff > 0 else "below"

    # Create output
    output_html = f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 30px; border-radius: 15px; color: white; text-align: center;'>
        <h2 style='margin: 0; font-size: 1.2em; opacity: 0.9;'>Predicted House Value</h2>
        <h1 style='margin: 10px 0; font-size: 3em; font-weight: bold;'>${pred_value:,}</h1>

        <div style='background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; margin: 20px 0;'>
            <p style='margin: 5px 0;'><strong>Confidence Range:</strong></p>
            <p style='margin: 5px 0; font-size: 1.1em;'>${conf_lower:,.0f} — ${conf_upper:,.0f}</p>
        </div>

        <div style='background: rgba(255,255,255,0.15); padding: 12px; border-radius: 10px;'>
            <p style='margin: 5px 0;'><strong>vs Dataset Average:</strong></p>
            <p style='margin: 5px 0; font-size: 1.2em;'>{abs(diff_pct):.1f}% {comparison} average (${avg_value:,.0f})</p>
        </div>

        <hr style='border: none; border-top: 1px solid rgba(255,255,255,0.3); margin: 20px 0;'>

        <div style='text-align: left; font-size: 0.9em; opacity: 0.9;'>
            <p><strong>📍 Location:</strong> {region} ({proximity})</p>
            <p><strong>💰 Income:</strong> ${income_dollars:,}/year</p>
            <p><strong>🗺️ Coordinates:</strong> {lat:.2f}°N, {abs(lon):.2f}°W</p>
        </div>
    </div>
    """

    return pn.pane.HTML(output_html, sizing_mode='stretch_width')

# ==============================================================================
# VISUALIZATION 5: Trust & Limitations Panel
# ==============================================================================

trust_panel = pn.pane.Markdown(f"""
## 🎯 Model Trust & Limitations

### ✅ **You CAN Trust This Model For:**
- **Typical properties** in the $100,000 - $400,000 range
- **General price estimates** for different California regions
- **Understanding** how income and location affect prices
- **Comparative analysis** between regions (e.g., SF Bay vs Central Valley)

### ⚠️ **Do NOT Trust This Model For:**
- **Luxury homes** > $500,000 (training data capped at $500k)
- **Individual property** valuations (use professional appraisal)
- **Extreme locations** outside typical California coordinates
- **Future predictions** (model trained on 1990 data)
- **Neighborhoods with unique characteristics** (schools, crime, amenities not included)

### 📊 **Model Performance Summary:**
- **R² Score:** {metrics['r2']:.3f} ({metrics['r2']*100:.1f}% of variance explained)
- **Typical Error (RMSE):** ${metrics['rmse']*100000:,.0f}
- **Average Error (MAE):** ${metrics['mae']*100000:,.0f}
- **Training Samples:** {metrics['n_samples']:,} block groups

### 🧠 **What the Model Learns:**
This Linear Regression model uses only **3 features**:
1. **Median Income** (+${metrics['coefficients']['median_income']*100000:,.0f} per $10k) — Most important
2. **Latitude** (${metrics['coefficients']['latitude']*100000:,.0f} per degree north) — North/South differences
3. **Longitude** (${metrics['coefficients']['longitude']*100000:,.0f} per degree east) — Coastal vs Inland

**What's Missing:** School quality, crime rates, walkability, local amenities, property condition,
lot size, number of bedrooms/bathrooms, and many other factors that affect real house prices.

### 💡 **Key Insight:**
The model explains **58% of price variation** with just income and location. The other **42%**
comes from factors not in our data. This shows that while location and income are important,
many other neighborhood and property-specific factors matter too!

""", width=800, styles={'background': '#f8f9fa', 'padding': '20px', 'border-radius': '10px'})

# ==============================================================================
# Build Dashboard Layout
# ==============================================================================

template = pn.template.FastListTemplate(
    title='🏠 California Housing Price Predictor (Option C)',
    sidebar=[
        pn.pane.Markdown("""
        ## 🎯 Try It Out!

        Adjust the controls below to predict house values for different scenarios.

        The model uses **Linear Regression** trained on 20,640 California properties
        from the 1990 Census.
        """),
        pn.pane.Markdown("### Input Parameters"),
        income_slider,
        region_selector,
        proximity_toggle,
        pn.pane.Markdown("""
        ---
        ### About This Model

        **Algorithm:** Linear Regression
        **Features:** Income, Latitude, Longitude
        **Performance:** 58% variance explained

        **Team 8 - CS 133 Final Sprint**
        """)
    ],
    main=[
        pn.pane.Markdown("## 🔮 Make a Prediction"),
        make_interactive_prediction,

        pn.pane.Markdown("---"),
        pn.pane.Markdown("## 📊 Model Visualizations"),

        pn.Tabs(
            ('Feature Importance', pn.pane.Plotly(create_feature_importance(), sizing_mode='stretch_width')),
            ('Actual vs Predicted', pn.pane.Plotly(create_actual_vs_predicted(), sizing_mode='stretch_width')),
            ('Residual Analysis', pn.pane.Plotly(create_residual_plot(), sizing_mode='stretch_width')),
            ('Trust & Limitations', trust_panel)
        )
    ],
    accent_base_color='#667eea',
    header_background='#4a5568'
)

# Make it servable
template.servable()

print("\n✅ Prediction dashboard ready!")
print("🌐 Open http://localhost:5006/prediction_app in your browser")
