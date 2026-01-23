import pandas as pd
import numpy as np
import joblib
from server import EnhancedTravelRiskPredictor

def run_test():
    """
    This is a simple test to verify the functionality of the
    EnhancedTravelRiskPredictor.predict_all_disasters method
    after the recent optimization.
    """
    try:
        # Initialize the predictor
        predictor = EnhancedTravelRiskPredictor()

        # Check if models are loaded
        if not predictor.disaster_models:
            print("No models loaded. Please train models first.")
            # As a workaround for this test, let's create dummy models
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            for disaster_type in ['flood', 'earthquake', 'landslide', 'cyclone', 'drought']:
                # Create a dummy model
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', RandomForestClassifier())
                ])
                # Fit with some dummy data to avoid NotFittedError
                X_dummy = np.random.rand(10, len(predictor.numerical_features) + len(predictor.categorical_features) -4)
                y_dummy = np.random.randint(0, 2, 10)

                # A bit of a hack to make the dummy model work
                # Get the preprocessor from a real model if it were available
                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import OneHotEncoder

                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), predictor.numerical_features),
                        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), predictor.categorical_features)
                    ])

                # Create a pipeline with the preprocessor and a classifier
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', RandomForestClassifier(random_state=42))
                ])

                # Create some dummy data that fits the preprocessor's expectations
                dummy_data = {
                    'Temperature_°C': [25], 'Humidity_': [70], 'Pressure_hPa': [1013],
                    'Wind_Speed_kmh': [10], 'Rainfall_mm': [5], 'Elevation_m': [100],
                    'Distance_to_River_km': [2], 'Population_Density': [1000],
                    'Water_Level_m': [2], 'River_Discharge_ms': [100],
                    'Historical_Events': [1], 'Latitude': [20], 'Longitude': [77],
                    'Seismic_Activity': [2], 'Slope_Angle': [5],
                    'Infrastructure': ['Medium'], 'Land_Cover': ['Urban'],
                    'Soil_Type': ['Loam'], 'Season': ['Summer']
                }
                X_dummy_df = pd.DataFrame(dummy_data)
                y_dummy = pd.Series([0])

                # Fit the pipeline
                pipeline.fit(X_dummy_df, y_dummy)

                predictor.disaster_models[disaster_type] = pipeline
            print("Dummy models created for testing.")


        # Create some dummy location data
        location_data = {
            'Temperature_°C': 25.0, 'Humidity_': 70.0, 'Pressure_hPa': 1013.0,
            'Wind_Speed_kmh': 10.0, 'Rainfall_mm': 0.0, 'Elevation_m': 100.0,
            'Distance_to_River_km': 2.0, 'Population_Density': 1000.0,
            'Water_Level_m': 2.0, 'River_Discharge_ms': 100.0,
            'Historical_Events': 1, 'Latitude': 20.0, 'Longitude': 77.0,
            'Seismic_Activity': 2.0, 'Slope_Angle': 5.0,
            'Infrastructure': 'Medium', 'Land_Cover': 'Urban',
            'Soil_Type': 'Loam', 'Season': 'Summer'
        }

        # Call the method
        predictions = predictor.predict_all_disasters(location_data)

        # Print the results
        if predictions:
            print("Successfully got predictions:")
            for disaster, result in predictions.items():
                print(f"  - {disaster.capitalize()}: Risk Level - {result['risk_level']}, Probability - {result['probability']:.2f}")
            print("\nTest passed!")
        else:
            print("Failed to get predictions.")

    except Exception as e:
        print(f"An error occurred during the test: {e}")

if __name__ == "__main__":
    run_test()
