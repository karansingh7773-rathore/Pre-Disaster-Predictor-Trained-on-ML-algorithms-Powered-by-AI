import pandas as pd
import numpy as np
import joblib
import json
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# AI Integration
try:
    from openai import OpenAI
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logger.warning("OpenAI library not found. AI recommendations will be disabled.")

class MultiDisasterModelTrainer:
    def __init__(self):
        """Initialize the multi-disaster model trainer"""
        self.models = {}
        self.categorical_features = ['Infrastructure', 'Land_Cover', 'Soil_Type', 'Season']
        self.numerical_features = [
            'Temperature_°C', 'Humidity_', 'Pressure_hPa', 'Wind_Speed_kmh',
            'Rainfall_mm', 'Elevation_m', 'Distance_to_River_km', 'Population_Density',
            'Water_Level_m', 'River_Discharge_ms', 'Historical_Events',
            'Latitude', 'Longitude', 'Seismic_Activity', 'Slope_Angle'
        ]

    def generate_synthetic_data_with_ai(self, disaster_type, n_samples=50):
        """Generates synthetic training data using a powerful generative AI."""
        logger.info(f"Using AI to generate {n_samples} synthetic data samples for '{disaster_type}'...")
        try:
            ai_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv('OPENROUTER_API_KEY')
            )

            # We need to provide all feature names to the AI
            all_features = self.numerical_features + self.categorical_features

            prompt = f"""
            Generate {n_samples} realistic and diverse data samples for training a '{disaster_type}' prediction model for locations in India.
            Use the following schema. Create varied, plausible scenarios based on known meteorological and geological patterns.

            SCHEMA: {all_features}
            TARGET: '{disaster_type.capitalize()}_Risk' (0 for no risk, 1 for risk)

            Example Scenarios:
            - For a 'flood', simulate heavy monsoon rainfall (e.g., >100mm) near a river in a low-elevation coastal city like Mumbai or Chennai.
            - For a 'drought', simulate a failed monsoon season (e.g., <5mm rainfall) with high temperatures in a semi-arid region like Jaipur.
            - For an 'earthquake', ensure 'Seismic_Activity' is high (e.g., > 4.0) for cities in known seismic zones like Delhi.

            IMPORTANT: Respond ONLY with a valid JSON array of objects. Do not include any other text, explanation, or markdown.
            """

            completion = ai_client.chat.completions.create(
                model="nvidia/nemotron-nano-9b-v2:free", # Use a powerful model for this
                messages=[
                    {"role": "system", "content": "You are a data science expert specializing in Indian climate and geology. You only output valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,
                max_tokens=4096
            )

            ai_response_str = completion.choices[0].message.content
            data_list = json.loads(ai_response_str)
            df = pd.DataFrame(data_list)

            logger.info(f"Successfully generated {len(df)} samples with AI.")
            return df

        except Exception as e:
            logger.error(f"AI data generation failed: {e}. Falling back to rule-based generation.")
            # Fallback to your original method if the API call fails
            return self.create_disaster_training_data(disaster_type, n_samples)

    def create_disaster_training_data(self, disaster_type, n_samples=1000):
        """Generate synthetic training data for different disasters"""
        np.random.seed(42)
        data = []

        for i in range(n_samples):
            # Base weather and geographic features
            temp = np.random.normal(25, 8)
            humidity = np.random.uniform(30, 100)
            pressure = np.random.normal(1013, 20)
            wind_speed = np.random.exponential(10)
            rainfall = np.random.exponential(5)
            elevation = np.random.uniform(0, 2000)
            distance_to_river = np.random.exponential(2)
            population_density = np.random.uniform(100, 15000)
            water_level = np.random.uniform(0.5, 8)
            river_discharge = np.random.uniform(10, 500)
            historical_events = np.random.randint(0, 10)
            seismic_activity = np.random.uniform(0, 8)
            slope_angle = np.random.uniform(0, 45)

            # Location (global coordinates)
            latitude = np.random.uniform(-60, 70)
            longitude = np.random.uniform(-180, 180)

            # Categorical features
            infrastructure = np.random.choice(['Poor', 'Medium', 'Good'], p=[0.3, 0.5, 0.2])
            land_cover = np.random.choice(['Urban', 'Forest', 'Agricultural', 'Water'], p=[0.4, 0.2, 0.3, 0.1])
            soil_type = np.random.choice(['Clay', 'Sand', 'Loam', 'Rock'], p=[0.3, 0.25, 0.35, 0.1])
            season = np.random.choice(['Spring', 'Summer', 'Monsoon', 'Winter'], p=[0.25, 0.25, 0.25, 0.25])

            # Calculate disaster-specific risk
            risk_score = self._calculate_disaster_risk(
                disaster_type, temp, humidity, pressure, wind_speed, rainfall,
                elevation, distance_to_river, population_density, historical_events,
                seismic_activity, slope_angle, infrastructure, land_cover, soil_type, season
            )

            # Convert to binary risk
            thresholds = {'flood': 6, 'earthquake': 4, 'landslide': 5, 'cyclone': 5, 'drought': 4}
            disaster_risk = 1 if risk_score >= thresholds.get(disaster_type, 5) else 0

            # Add randomness
            if np.random.random() < 0.1:
                disaster_risk = 1 - disaster_risk

            data.append({
                'Temperature_°C': round(temp, 2),
                'Humidity_': round(humidity, 2),
                'Pressure_hPa': round(pressure, 2),
                'Wind_Speed_kmh': round(wind_speed, 2),
                'Rainfall_mm': round(rainfall, 2),
                'Elevation_m': round(elevation, 2),
                'Distance_to_River_km': round(distance_to_river, 2),
                'Population_Density': round(population_density, 2),
                'Water_Level_m': round(water_level, 2),
                'River_Discharge_ms': round(river_discharge, 2),
                'Historical_Events': historical_events,
                'Latitude': round(latitude, 4),
                'Longitude': round(longitude, 4),
                'Seismic_Activity': round(seismic_activity, 2),
                'Slope_Angle': round(slope_angle, 2),
                'Infrastructure': infrastructure,
                'Land_Cover': land_cover,
                'Soil_Type': soil_type,
                'Season': season,
                f'{disaster_type.capitalize()}_Risk': disaster_risk
            })

        return pd.DataFrame(data)

    def _calculate_disaster_risk(self, disaster_type, temp, humidity, pressure, wind_speed,
                               rainfall, elevation, distance_to_river, population_density,
                               historical_events, seismic_activity, slope_angle,
                               infrastructure, land_cover, soil_type, season):
        """Calculate risk score based on disaster type"""
        risk_score = 0

        if disaster_type == 'flood':
            if rainfall > 15: risk_score += 3
            elif rainfall > 8: risk_score += 2
            if elevation < 50: risk_score += 2
            if distance_to_river < 1: risk_score += 3
            if historical_events > 5: risk_score += 2
            if infrastructure == 'Poor': risk_score += 2
            if soil_type == 'Clay': risk_score += 1

        elif disaster_type == 'earthquake':
            if seismic_activity > 5: risk_score += 4
            elif seismic_activity > 3: risk_score += 2
            if infrastructure == 'Poor': risk_score += 3
            if population_density > 10000: risk_score += 1
            if historical_events > 3: risk_score += 2

        elif disaster_type == 'landslide':
            if slope_angle > 25: risk_score += 3
            elif slope_angle > 15: risk_score += 2
            if rainfall > 20: risk_score += 3
            if soil_type == 'Clay': risk_score += 2
            if land_cover == 'Forest': risk_score -= 1
            if elevation > 500: risk_score += 1

        elif disaster_type == 'cyclone':
            if wind_speed > 60: risk_score += 4
            elif wind_speed > 40: risk_score += 2
            if pressure < 990: risk_score += 3
            if elevation < 20: risk_score += 2
            if temp > 26 and humidity > 80: risk_score += 2

        elif disaster_type == 'drought':
            if rainfall < 2: risk_score += 3
            elif rainfall < 5: risk_score += 2
            if temp > 35: risk_score += 3
            elif temp > 30: risk_score += 1
            if humidity < 40: risk_score += 2
            if season == 'Summer': risk_score += 1

        return risk_score

    def train_disaster_models(self):
        """Train models for all disaster types with optional AI-generated data"""
        disaster_types = ['flood', 'earthquake', 'landslide', 'cyclone', 'drought']

        for disaster_type in disaster_types:
            logger.info(f"Training {disaster_type} prediction model...")

            # Use AI-generated data if available (smaller sample size due to API limits)
            # df = self.generate_synthetic_data_with_ai(disaster_type, n_samples=200)

            # For demo purposes, use traditional method for faster training
            df = self.create_disaster_training_data(disaster_type, n_samples=2000)

            X = df.drop(f'{disaster_type.capitalize()}_Risk', axis=1)
            y = df[f'{disaster_type.capitalize()}_Risk']

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), self.numerical_features),
                    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), self.categorical_features)
                ])

            model = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2
                ))
            ])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            logger.info(f"{disaster_type.capitalize()} model accuracy: {accuracy:.3f}")

            self.models[disaster_type] = model
            joblib.dump(model, f'{disaster_type}_model.pkl')
            df.to_csv(f'{disaster_type}_training_data.csv', index=False)

        logger.info("All disaster models trained successfully!")
        return self.models
