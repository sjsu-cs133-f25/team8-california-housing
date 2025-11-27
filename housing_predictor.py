"""
California Housing Price Predictor
Simple Linear Regression model to predict house values based on income and location.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.datasets import fetch_california_housing


class HousingPredictor:
    """
    Predicts California housing prices using Linear Regression.

    Features:
        - median_income: Household income in $10k units
        - latitude: Geographic latitude
        - longitude: Geographic longitude

    Target:
        - median_house_value: House value in $100k units
    """

    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False
        self.metrics = {}
        self.feature_names = ['median_income', 'latitude', 'longitude']

    def train(self, df=None):
        """
        Train the model on California housing data.

        Args:
            df: Optional DataFrame. If None, loads from sklearn.

        Returns:
            dict: Model performance metrics
        """
        # Load data if not provided
        if df is None:
            housing = fetch_california_housing(as_frame=True)
            df = housing.frame
            df.columns = [
                'median_income', 'housing_median_age', 'avg_rooms',
                'avg_bedrooms', 'population', 'avg_occupancy',
                'latitude', 'longitude', 'median_house_value'
            ]

        # Prepare features and target
        X = df[self.feature_names].values
        y = df['median_house_value'].values

        # Train model
        self.model.fit(X, y)
        self.is_trained = True

        # Calculate metrics
        y_pred = self.model.predict(X)
        self.metrics = {
            'r2': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'n_samples': len(y)
        }

        # Store coefficients for interpretation
        self.metrics['coefficients'] = {
            'median_income': self.model.coef_[0],
            'latitude': self.model.coef_[1],
            'longitude': self.model.coef_[2],
            'intercept': self.model.intercept_
        }

        return self.metrics

    def predict(self, median_income, latitude, longitude):
        """
        Predict house value for given inputs.

        Args:
            median_income: float, household income in $10k units (e.g., 5.0 = $50k)
            latitude: float, geographic latitude (32-42 for California)
            longitude: float, geographic longitude (-124 to -114 for California)

        Returns:
            dict: {
                'predicted_value': float (in $100k units),
                'predicted_value_dollars': int (in dollars),
                'confidence_lower': float,
                'confidence_upper': float
            }
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions. Call train() first.")

        # Prepare input
        X = np.array([[median_income, latitude, longitude]])

        # Make prediction
        predicted = self.model.predict(X)[0]

        # Calculate confidence interval (±1 RMSE as simple approximation)
        rmse = self.metrics['rmse']
        confidence_lower = max(0, predicted - rmse)
        confidence_upper = predicted + rmse

        return {
            'predicted_value': predicted,
            'predicted_value_dollars': int(predicted * 100_000),
            'confidence_lower': confidence_lower * 100_000,
            'confidence_upper': confidence_upper * 100_000,
            'confidence_lower_100k': confidence_lower,
            'confidence_upper_100k': confidence_upper
        }

    def get_feature_importance(self):
        """
        Get human-readable feature importance.

        Returns:
            str: Formatted string explaining coefficient impacts
        """
        if not self.is_trained:
            return "Model not trained yet"

        coef = self.metrics['coefficients']

        # Convert to dollar impacts
        income_impact = coef['median_income'] * 100_000  # per $10k income
        lat_impact = coef['latitude'] * 100_000  # per degree latitude
        lon_impact = coef['longitude'] * 100_000  # per degree longitude

        explanation = f"""
Model Feature Impacts:
• Income: ${income_impact:,.0f} per +$10k household income
• Latitude: ${lat_impact:,.0f} per +1° north
• Longitude: ${lon_impact:,.0f} per +1° east
• Base value: ${coef['intercept']*100_000:,.0f}
        """.strip()

        return explanation

    def get_metrics_summary(self):
        """
        Get formatted model performance summary.

        Returns:
            str: Human-readable metrics
        """
        if not self.is_trained:
            return "Model not trained yet"

        m = self.metrics
        summary = f"""
Model Performance:
• R² Score: {m['r2']:.3f} ({m['r2']*100:.1f}% variance explained)
• RMSE: ${m['rmse']*100_000:,.0f}
• MAE: ${m['mae']*100_000:,.0f}
• Training samples: {m['n_samples']:,}
        """.strip()

        return summary


# Convenience function for quick predictions
def predict_house_value(median_income, latitude, longitude):
    """
    Quick prediction function (trains model if needed).

    Args:
        median_income: Household income in $10k units
        latitude: Geographic latitude
        longitude: Geographic longitude

    Returns:
        int: Predicted house value in dollars
    """
    predictor = HousingPredictor()
    predictor.train()
    result = predictor.predict(median_income, latitude, longitude)
    return result['predicted_value_dollars']


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("California Housing Price Predictor")
    print("=" * 60)

    # Create and train predictor
    predictor = HousingPredictor()
    print("\nTraining model...")
    metrics = predictor.train()

    print("\n" + predictor.get_metrics_summary())
    print("\n" + predictor.get_feature_importance())

    # Example predictions
    print("\n" + "=" * 60)
    print("Example Predictions:")
    print("=" * 60)

    examples = [
        {"income": 5.0, "lat": 37.8, "lon": -122.4, "desc": "San Francisco Bay Area, $50k income"},
        {"income": 10.0, "lat": 34.0, "lon": -118.2, "desc": "Los Angeles, $100k income"},
        {"income": 3.0, "lat": 36.7, "lon": -119.8, "desc": "Central Valley, $30k income"}
    ]

    for ex in examples:
        result = predictor.predict(ex['income'], ex['lat'], ex['lon'])
        print(f"\n{ex['desc']}:")
        print(f"  Predicted: ${result['predicted_value_dollars']:,}")
        print(f"  Range: ${result['confidence_lower']:,.0f} - ${result['confidence_upper']:,.0f}")
