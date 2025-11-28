"""
California Housing Price Predictor
Simple Linear Regression model to predict house values based on income and location.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import fetch_california_housing


class HousingPredictor:
    """
    Predicts California housing prices using Linear Regression.

    Features (simple mode - 3):
        - median_income: Household income in $10k units
        - latitude: Geographic latitude
        - longitude: Geographic longitude

    Features (full mode - 8):
        - All above plus:
        - housing_median_age: Median age of houses
        - avg_rooms: Average number of rooms
        - avg_bedrooms: Average number of bedrooms
        - population: Block group population
        - avg_occupancy: Average occupancy

    Target:
        - median_house_value: House value in $100k units
    """

    def __init__(self, use_all_features=False, model_type='linear'):
        """
        Initialize the predictor.

        Args:
            use_all_features: If True, uses all 8 features. If False, uses 3 features (default).
            model_type: 'linear' for Linear Regression (default) or 'random_forest' for Random Forest
        """
        self.model_type = model_type
        self.use_all_features = use_all_features
        self.is_trained = False
        self.metrics = {}

        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1  # Use all CPU cores
            )
        elif model_type == 'linear':
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'linear' or 'random_forest'")

        # Set feature names based on mode
        if use_all_features:
            self.feature_names = ['median_income', 'housing_median_age', 'avg_rooms',
                                 'avg_bedrooms', 'population', 'avg_occupancy',
                                 'latitude', 'longitude']
        else:
            self.feature_names = ['median_income', 'latitude', 'longitude']

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_means = None  # Store means for prediction with all features

    def train(self, df=None, test_size=0.2, random_state=42):
        """
        Train the model on California housing data with train/test split.

        Args:
            df: Optional DataFrame. If None, loads from sklearn.
            test_size: Fraction of data to use for testing (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)

        Returns:
            dict: Model performance metrics (both training and test)
        """
        # Load data if not provided
        if df is None:
            housing = fetch_california_housing(as_frame=True)
            df = housing.frame
            # Rename columns to match our feature names
            df.columns = [
                'median_income', 'housing_median_age', 'avg_rooms',
                'avg_bedrooms', 'population', 'avg_occupancy',
                'latitude', 'longitude', 'median_house_value'
            ]

        # Prepare features and target
        X = df[self.feature_names].values
        y = df['median_house_value'].values

        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Store feature means for prediction (used when use_all_features=True)
        self.feature_means = np.mean(self.X_train, axis=0)

        # Train model on training set only
        self.model.fit(self.X_train, self.y_train)
        self.is_trained = True

        # Calculate training metrics
        y_train_pred = self.model.predict(self.X_train)
        train_metrics = {
            'train_r2': r2_score(self.y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'train_mae': mean_absolute_error(self.y_train, y_train_pred),
        }

        # Calculate test metrics (on unseen data)
        y_test_pred = self.model.predict(self.X_test)
        test_metrics = {
            'test_r2': r2_score(self.y_test, y_test_pred),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'test_mae': mean_absolute_error(self.y_test, y_test_pred),
        }

        # Store all metrics
        self.metrics = {
            **train_metrics,
            **test_metrics,
            'n_train_samples': len(self.y_train),
            'n_test_samples': len(self.y_test),
            'n_total_samples': len(y)
        }

        # Store coefficients for interpretation (only for linear models)
        if self.model_type == 'linear':
            self.metrics['coefficients'] = {
                'median_income': self.model.coef_[0],
                'latitude': self.model.coef_[1 if not self.use_all_features else 6],
                'longitude': self.model.coef_[2 if not self.use_all_features else 7],
                'intercept': self.model.intercept_
            }

        # Legacy metrics for backward compatibility (use test metrics)
        self.metrics['r2'] = test_metrics['test_r2']
        self.metrics['rmse'] = test_metrics['test_rmse']
        self.metrics['mae'] = test_metrics['test_mae']
        self.metrics['n_samples'] = self.metrics['n_test_samples']

        return self.metrics

    def cross_validate(self, df=None, cv=5):
        """
        Perform k-fold cross-validation to assess model stability.

        Args:
            df: Optional DataFrame. If None, loads from sklearn.
            cv: Number of cross-validation folds (default: 5)

        Returns:
            dict: Cross-validation results with mean and std of scores
        """
        # Load data if not provided
        if df is None:
            housing = fetch_california_housing(as_frame=True)
            df = housing.frame
            # Rename columns to match our feature names
            df.columns = [
                'median_income', 'housing_median_age', 'avg_rooms',
                'avg_bedrooms', 'population', 'avg_occupancy',
                'latitude', 'longitude', 'median_house_value'
            ]

        # Prepare features and target
        X = df[self.feature_names].values
        y = df['median_house_value'].values

        # Create a fresh model instance for cross-validation
        if self.model_type == 'random_forest':
            cv_model = RandomForestRegressor(
                n_estimators=100, max_depth=20, min_samples_split=5,
                random_state=42, n_jobs=-1
            )
        else:
            cv_model = LinearRegression()

        # Perform cross-validation for multiple metrics
        cv_r2 = cross_val_score(cv_model, X, y, cv=cv, scoring='r2')
        cv_neg_mse = cross_val_score(cv_model, X, y, cv=cv, scoring='neg_mean_squared_error')
        cv_neg_mae = cross_val_score(cv_model, X, y, cv=cv, scoring='neg_mean_absolute_error')

        # Convert negative scores to positive and calculate RMSE
        cv_rmse = np.sqrt(-cv_neg_mse)
        cv_mae = -cv_neg_mae

        cv_results = {
            'cv_r2_scores': cv_r2,
            'cv_r2_mean': cv_r2.mean(),
            'cv_r2_std': cv_r2.std(),
            'cv_rmse_scores': cv_rmse,
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'cv_mae_scores': cv_mae,
            'cv_mae_mean': cv_mae.mean(),
            'cv_mae_std': cv_mae.std(),
            'cv_folds': cv
        }

        return cv_results

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

        # Prepare input based on feature mode
        if self.use_all_features:
            # For 8-feature models, use training means for missing features
            # Order: median_income, housing_median_age, avg_rooms, avg_bedrooms,
            #        population, avg_occupancy, latitude, longitude
            X = np.array([[
                median_income,                # User provided
                self.feature_means[1],        # Use mean housing age
                self.feature_means[2],        # Use mean rooms
                self.feature_means[3],        # Use mean bedrooms
                self.feature_means[4],        # Use mean population
                self.feature_means[5],        # Use mean occupancy
                latitude,                     # User provided
                longitude                     # User provided
            ]])
        else:
            # For 3-feature models, use only income, lat, lon
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

    def predict_with_features(self, median_income, latitude, longitude,
                              housing_age=None, avg_rooms=None, avg_bedrooms=None,
                              population=None, avg_occupancy=None):
        """
        Predict house value with optional feature inputs.

        This method allows you to specify property-specific features (age, rooms, bedrooms)
        while using training dataset means for unspecified features.

        Args:
            median_income: float, household income in $10k units (e.g., 5.0 = $50k)
            latitude: float, geographic latitude (32-42 for California)
            longitude: float, geographic longitude (-124 to -114 for California)
            housing_age: float, optional, median age of houses in years (default: training mean)
            avg_rooms: float, optional, average number of rooms (default: training mean)
            avg_bedrooms: float, optional, average number of bedrooms (default: training mean)
            population: float, optional, block group population (default: training mean)
            avg_occupancy: float, optional, average occupancy (default: training mean)

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

        # Only works with 8-feature models
        if not self.use_all_features:
            raise ValueError("predict_with_features() requires use_all_features=True. "
                           "Use predict() for 3-feature models.")

        # Build feature array with user inputs or training means
        # Order: median_income, housing_median_age, avg_rooms, avg_bedrooms,
        #        population, avg_occupancy, latitude, longitude
        X = np.array([[
            median_income,
            housing_age if housing_age is not None else self.feature_means[1],
            avg_rooms if avg_rooms is not None else self.feature_means[2],
            avg_bedrooms if avg_bedrooms is not None else self.feature_means[3],
            population if population is not None else self.feature_means[4],
            avg_occupancy if avg_occupancy is not None else self.feature_means[5],
            latitude,
            longitude
        ]])

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

    def get_feature_importance_rf(self):
        """
        Get feature importance for Random Forest models.

        Returns:
            list: List of (feature_name, importance) tuples sorted by importance
        """
        if not self.is_trained:
            return []

        if self.model_type != 'random_forest':
            return []

        if not hasattr(self.model, 'feature_importances_'):
            return []

        importances = self.model.feature_importances_
        feature_importance = list(zip(self.feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        return feature_importance


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
