#AI_chatbot.py
# Enhanced Multi-Disaster and Travel Risk Prediction System with Advanced AI Integration
import pandas as pd
import numpy as np
import joblib
import json
import os
import logging
from typing import Optional, Dict
import warnings
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Import refactored modules
from data_fetchers import IMDDataFetcher, EnhancedDataFetcher, NetworkConnectivityChecker
from ai_engine import DisasterChatbot, AIGuidanceSystem
from ml_training import MultiDisasterModelTrainer

warnings.filterwarnings('ignore')

# Load environment variables at the top
load_dotenv()

# AI Integration
try:
    from openai import OpenAI
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logger.warning("OpenAI library not found. AI recommendations will be disabled.")

class TravelRiskPredictor:
    def __init__(self):
        """Initialize travel risk predictor"""
        self.disaster_models = {}
        self.load_models()
        
        self.numerical_features = [
            'Temperature_°C', 'Humidity_', 'Pressure_hPa', 'Wind_Speed_kmh',
            'Rainfall_mm', 'Elevation_m', 'Distance_to_River_km', 'Population_Density',
            'Water_Level_m', 'River_Discharge_ms', 'Historical_Events',
            'Latitude', 'Longitude', 'Seismic_Activity', 'Slope_Angle'
        ]
        self.categorical_features = ['Infrastructure', 'Land_Cover', 'Soil_Type', 'Season']
        self.all_features = self.numerical_features + self.categorical_features
        
    def load_models(self):
        """Load all disaster prediction models"""
        disaster_types = ['flood', 'earthquake', 'landslide', 'cyclone', 'drought']
        
        for disaster_type in disaster_types:
            try:
                self.disaster_models[disaster_type] = joblib.load(f'{disaster_type}_model.pkl')
                logger.info(f"Model loaded: {disaster_type.capitalize()}")
            except FileNotFoundError:
                logger.error(f"Model not found: {disaster_type.capitalize()}")
    
    def predict_all_disasters(self, location_data):
        """Predict all disaster risks for a location"""
        if not self.disaster_models:
            logger.error("No models loaded. Please train models first.")
            return None
        
        try:
            df = pd.DataFrame([location_data])
            
            for feature in self.all_features:
                if feature not in df.columns:
                    df[feature] = self._get_default_value(feature)
            
            df = df[self.all_features]
            
            results = {}
            
            for disaster_type, model in self.disaster_models.items():
                prediction = model.predict(df)[0]
                probability = model.predict_proba(df)[0, 1]
                
                results[disaster_type] = {
                    'risk': bool(prediction),
                    'probability': float(probability),
                    'risk_level': self._get_risk_level(probability)
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None
    
    def _display_prediction_results(self, predictions, location_data):
        """Display prediction results in formatted way"""
        print("DISASTER RISK ASSESSMENT:")
        
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1]['probability'], reverse=True)
        
        for disaster, result in sorted_predictions:
            risk_indicator = "🚨" if result['risk_level'] == 'High' else "⚠️" if result['risk_level'] == 'Medium' else "✅"
            print(f"  {risk_indicator} {disaster.upper():12}: {result['risk_level']:6} ({result['probability']:.1%})")
        
        print(f"\nCURRENT CONDITIONS:")
        print(f"  Temperature: {location_data.get('Temperature_°C', 'N/A')}°C")
        print(f"  Humidity: {location_data.get('Humidity_', 'N/A')}%")
        print(f"  Rainfall: {location_data.get('Rainfall_mm', 'N/A')}mm")
        print(f"  Wind Speed: {location_data.get('Wind_Speed_kmh', 'N/A')} km/h")
    
    def _get_default_value(self, feature):
        """Get default values for missing features"""
        defaults = {
            'Temperature_°C': 25.0, 'Humidity_': 70.0, 'Pressure_hPa': 1013.0,
            'Wind_Speed_kmh': 10.0, 'Rainfall_mm': 0.0, 'Elevation_m': 100.0,
            'Distance_to_River_km': 2.0, 'Population_Density': 1000.0,
            'Water_Level_m': 2.0, 'River_Discharge_ms': 100.0,
            'Historical_Events': 1, 'Latitude': 20.0, 'Longitude': 77.0,
            'Seismic_Activity': 2.0, 'Slope_Angle': 5.0,
            'Infrastructure': 'Medium', 'Land_Cover': 'Urban',
            'Soil_Type': 'Loam', 'Season': 'Summer'
        }
        return defaults.get(feature, 0.0)
    
    def _get_risk_level(self, probability):
        """Convert probability to risk level"""
        if probability < 0.3:
            return "Low"
        elif probability < 0.7:
            return "Medium"
        else:
            return "High"
            
class EnhancedTravelRiskPredictor(TravelRiskPredictor):
    """Enhanced predictor with IMD integration and advanced AI features"""
    
    def __init__(self):
        super().__init__()
        self.imd_fetcher = IMDDataFetcher()
        self.ai_guidance = AIGuidanceSystem()
        self.chatbot = DisasterChatbot()
    
    def handle_what_if_scenario(self, original_data, city, user_query):
        """Handles 'what-if' scenarios from the user using AI."""
        logger.info("Simulating 'what-if' scenario...")
        try:
            if not hasattr(self, 'ai_client'):
                self.ai_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv('OPENROUTER_API_KEY')
                )

            analysis_prompt = (
                f"Analyze this what-if scenario:\n"
                f"Location: {city}\n"
                f"Query: {user_query}\n"
                f"Current Data: {json.dumps(original_data)}\n\n"
                "Provide response in this JSON format:\n"
                "{\n"
                '    "changes": [\n'
                '        {\n'
                '            "parameter": "parameter_name",\n'
                '            "value": numeric_value\n'
                "        }\n"
                "    ],\n"
                '    "summary": "description of changes"\n'
                "}"
            )

            response = self.ai_client.chat.completions.create(
                model="deepseek/deepseek-chat-v3.1:free",
                messages=[
                    {"role": "system", "content": "You are a disaster risk analysis expert. Analyze what-if scenarios and provide structured responses in JSON format."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            ai_analysis = json.loads(content)

            modified_data = original_data.copy()
            if "changes" in ai_analysis:
                for change in ai_analysis["changes"]:
                    param = change.get("parameter")
                    value = change.get("value")
                    if param in modified_data and value is not None:
                        try:
                            modified_data[param] = float(value)
                        except (ValueError, TypeError):
                            logger.warning(f"Warning: Could not convert value for {param}")
                            continue

            original_predictions = self.predict_all_disasters(original_data)
            new_predictions = self.predict_all_disasters(modified_data)

            response = f"\nWHAT-IF ANALYSIS FOR {city.upper()}\n"
            response += "=" * 50 + "\n\n"
            
            response += "SCENARIO SUMMARY:\n"
            response += ai_analysis.get("summary", "Analysis not available") + "\n\n"
            
            response += "PARAMETER CHANGES:\n"
            for change in ai_analysis.get("changes", []):
                param = change.get("parameter")
                old_val = original_data.get(param, "N/A")
                new_val = modified_data.get(param, "N/A")
                response += f"• {param}: {old_val} → {new_val}\n"
            
            response += "\nRISK IMPACT ANALYSIS:\n"
            for disaster in new_predictions.keys():
                old_prob = original_predictions[disaster]['probability']
                new_prob = new_predictions[disaster]['probability']
                change = new_prob - old_prob
                arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
                response += f"• {disaster.capitalize()} Risk: {old_prob:.1%} → {new_prob:.1%} {arrow}\n"

            return response

        except Exception as e:
            logger.error(f"Error in what-if analysis: {str(e)}")
            return self._handle_what_if_fallback(original_data, city, user_query)
            
    
    def _handle_what_if_fallback(self, original_data, city, user_query):
        """Fallback method for what-if analysis when AI processing fails"""
        try:
            # Simple keyword-based parameter adjustments
            modified_data = original_data.copy()
            keywords = {
                'rain': {'param': 'Rainfall_mm', 'increase': 50, 'decrease': -30},
                'temperature': {'param': 'Temperature_°C', 'increase': 5, 'decrease': -5},
                'wind': {'param': 'Wind_Speed_kmh', 'increase': 20, 'decrease': -10},
                'water': {'param': 'Water_Level_m', 'increase': 1, 'decrease': -0.5},
            }
            
            # Check for keywords and modifiers
            for key, info in keywords.items():
                if key in user_query.lower():
                    param = info['param']
                    if 'increase' in user_query.lower() or 'higher' in user_query.lower():
                        modified_data[param] = original_data.get(param, 0) + info['increase']
                    elif 'decrease' in user_query.lower() or 'lower' in user_query.lower():
                        modified_data[param] = original_data.get(param, 0) + info['decrease']
            
            # Get predictions
            original_predictions = self.predict_all_disasters(original_data)
            modified_predictions = self.predict_all_disasters(modified_data)
            
            # Format response
            response = f"\nBASIC WHAT-IF ANALYSIS FOR {city.upper()}\n"
            response += "=" * 50 + "\n\n"
            
            response += "CHANGES DETECTED:\n"
            for param, new_val in modified_data.items():
                old_val = original_data.get(param)
                if old_val != new_val:
                    response += f"• {param}: {old_val} → {new_val}\n"
            
            response += "\nRISK CHANGES:\n"
            for disaster in modified_predictions.keys():
                old_prob = original_predictions[disaster]['probability']
                new_prob = modified_predictions[disaster]['probability']
                change = new_prob - old_prob
                arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
                response += f"• {disaster.capitalize()} Risk: {old_prob:.1%} → {new_prob:.1%} {arrow}\n"
            
            return response
            
        except Exception as e:
            return f"Could not process what-if scenario: {str(e)}\nPlease try with more specific parameters."

    def predict_with_imd_integration(self, location_data, city_name):
        """Enhanced prediction with IMD data integration"""
        if not self.disaster_models:
            logger.error("No models loaded. Please train models first.")
            return None
        
        try:
            # Get IMD data
            logger.info("Fetching IMD official data...")
            imd_data = self.imd_fetcher.get_comprehensive_imd_data(city_name)
            
            # Make standard predictions
            predictions = self.predict_all_disasters(location_data)
            
            if predictions:
                print(f"\nENHANCED PREDICTION RESULTS FOR {city_name.upper()}")
                print("="*60)
                
                # Display IMD alert status
                if imd_data.get('risk_indicators'):
                    alert_level = imd_data['risk_indicators'].get('overall_alert_level', 'green')
                    print(f"IMD ALERT STATUS: {alert_level.upper()}")
                    
                    high_warnings = imd_data['risk_indicators'].get('high_risk_warnings', [])
                    if high_warnings:
                        print("OFFICIAL WARNINGS (Simplified):")
                        for warning in high_warnings:
                            raw_text = warning.get('description', warning.get('warning_type', 'Weather Warning'))
                            summary = self.imd_fetcher.summarize_imd_warning(raw_text)
                            print(f"  ⚠️  {summary['summary_en']}")
                    print()
                
                # Display predictions
                self._display_prediction_results(predictions, location_data)
                
                # Get enhanced AI guidance
                print("\nENHANCED AI SAFETY GUIDANCE")
                print("="*50)
                guidance = self.ai_guidance.get_intelligent_guidance(
                    city_name, {'risks': predictions}, imd_data
                )
                print(guidance)
                
                # Ask if user wants explanation
                print("\n" + "="*50)
                explain = input("Want detailed explanation of these predictions? (y/n): ").strip().lower()
                if explain in ['y', 'yes']:
                    print("\nPREDICTION EXPLANATION:")
                    print("-" * 40)
                    explanation = self.chatbot.explain_prediction(predictions, city_name, location_data)
                    print(explanation)
                
                return {
                    'predictions': predictions,
                    'imd_data': imd_data,
                    'ai_guidance': guidance,
                    'city': city_name
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in enhanced prediction: {e}")
            return None

# Enhanced main execution functions with advanced AI integration

def interactive_disaster_chat_mode():
    """New conversational mode for disaster preparedness with advanced NLU"""
    print("\nINTERACTIVE DISASTER PREPAREDNESS CHAT")
    print("-" * 50)
    
    chatbot = DisasterChatbot()
    
    print("Welcome to your personal disaster preparedness advisor!")
    print("I can help with emergency planning, safety tips, and answer questions about disasters.")
    print("\nSample questions you can ask:")
    print("• 'What should I pack in my emergency kit?'")
    print("• 'How do I prepare for a flood?'")
    print("• 'What's the difference between a watch and a warning?'")
    print("• 'How do I secure my documents before a cyclone?'")
    print("• 'What should I do during an earthquake?'")
    
    city = input("\nOptional - Enter your city for localized advice: ").strip()
    
    chatbot.interactive_chat_mode(city if city else None)

def predict_disaster_with_enhanced_ai_guidance(city):
    """Enhanced prediction with IMD integration and advanced AI features"""
    print(f"\nENHANCED DISASTER PREDICTION FOR {city.upper()}")
    print("="*60)
    
    predictor = EnhancedTravelRiskPredictor()
    data_fetcher = EnhancedDataFetcher()
    
    if not predictor.disaster_models:
        print("No models loaded. Please train models first.")
        return
    
    # Get location data
    print("Fetching comprehensive data (Weather + IMD)...")
    location_data = data_fetcher.get_comprehensive_location_data(city)
    
    if not location_data:
        print(f"Could not fetch data for {city}")
        return
    
    # Make enhanced predictions
    result = predictor.predict_with_imd_integration(location_data, city)
    
    if result:
        # Store latest data for what-if scenarios
        latest_location_data = location_data
        
        # Offer follow-up options
        print("\n" + "="*50)
        print("FOLLOW-UP OPTIONS:")
        print("1. Chat about disaster preparedness")
        print("2. Ask 'what-if' scenario questions")
        print("3. Get more detailed explanations")
        print("4. Exit")
        
        while True:
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '4':
                print("\nStay safe! Remember to keep your emergency kit updated.")
                break
            
            if choice == '1':
                predictor.chatbot.interactive_chat_mode(city, result['predictions'])
                break
            elif choice == '2':
                what_if_query = input("Ask your what-if question: ").strip()
                if what_if_query:
                    response = predictor.handle_what_if_scenario(latest_location_data, city, what_if_query)
                    print(f"\nWhat-if Analysis:\n{response}")
                    continue
                else:
                    print("Please enter a valid what-if question.")
                    continue
            elif choice == '3':
                explanation = predictor.chatbot.explain_prediction(result['predictions'], city, location_data)
                print(f"\nDetailed Explanation:\n{explanation}")
                continue
            elif choice == '4':
                break
            else:
                print("Invalid choice. Please select 1-4.")
                continue
    
    return result

def enhanced_future_forecast_with_ai_guidance(city, days=7):
    """Enhanced future predictions with IMD integration and AI analysis"""
    print(f"\nENHANCED FUTURE FORECAST FOR {city.upper()}")
    print(f"Forecast Period: {days} days")
    print("="*60)
    
    predictor = EnhancedTravelRiskPredictor()
    data_fetcher = EnhancedDataFetcher()
    
    if not predictor.disaster_models:
        print("No models loaded. Please train models first.")
        return
    
    # Get IMD data for current conditions
    print("Fetching IMD official data...")
    imd_data = predictor.imd_fetcher.get_comprehensive_imd_data(city)
    
    # Get forecast data
    print("Fetching forecast data...")
    forecast_data = data_fetcher.get_forecast_data(city, days)
    
    if not forecast_data:
        print(f"Could not fetch forecast data for {city}")
        return
    
    # Make predictions for each day
    future_predictions = []
    
    for day_data in forecast_data:
        daily_predictions = predictor.predict_all_disasters(day_data)
        if daily_predictions:
            future_predictions.append({
                'date': day_data['date'],
                'weather': {
                    'temp': day_data['Temperature_°C'],
                    'rainfall': day_data['Rainfall_mm'],
                    'wind': day_data['Wind_Speed_kmh'],
                    'condition': day_data['Weather_Condition']
                },
                'risks': daily_predictions
            })
    
    if future_predictions:
        # Display IMD current status
        if imd_data.get('risk_indicators'):
            alert_level = imd_data['risk_indicators'].get('overall_alert_level', 'green')
            print(f"CURRENT IMD ALERT STATUS: {alert_level.upper()}")
            
            high_warnings = imd_data['risk_indicators'].get('high_risk_warnings', [])
            if high_warnings:
                print("ACTIVE OFFICIAL WARNINGS:")
                for warning in high_warnings:
                    print(f"  Warning: {warning.get('warning_type', 'Weather Warning')}")
            print()
        
        # Display forecast results
        print("ENHANCED FUTURE RISK FORECAST:")
        print("-" * 50)
        
        for pred in future_predictions:
            print(f"\nDate: {pred['date']}")
            weather = pred['weather']
            print(f"Weather: {weather['temp']}°C, {weather['condition']}")
            print(f"Rainfall: {weather['rainfall']}mm | Wind: {weather['wind']} km/h")
            
            # Show risks with priority
            sorted_risks = sorted(pred['risks'].items(), key=lambda x: x[1]['probability'], reverse=True)
            high_risks = [disaster for disaster, risk in sorted_risks if risk['risk_level'] == 'High']
            medium_risks = [disaster for disaster, risk in sorted_risks if risk['risk_level'] == 'Medium']
            
            if high_risks:
                print(f"HIGH RISKS: {', '.join(high_risks)}")
            elif medium_risks:
                print(f"MEDIUM RISKS: {', '.join(medium_risks)}")
            else:
                print("Risk Level: Normal")
            print("-" * 40)
        
        # Get comprehensive AI guidance with IMD integration
        print("\nENHANCED AI FORECAST GUIDANCE")
        print("="*50)
        guidance = predictor.ai_guidance.get_intelligent_guidance(city, future_predictions, imd_data)
        print(guidance)
        print("="*50)
        
        # Offer interactive options
        print("\nFOLLOW-UP OPTIONS:")
        print("1. Discuss these forecasts in detail")
        print("2. Ask specific questions about the forecast")
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        if choice == '1':
            # Create context from forecast
            high_risk_days = sum(1 for pred in future_predictions 
                               for disaster, risk in pred['risks'].items() 
                               if risk['risk_level'] == 'High')
            
            context_info = None
            if high_risk_days > 0:
                context_info = f"Forecast shows {high_risk_days} high-risk disaster predictions over {days} days"
            
            predictor.chatbot.interactive_chat_mode(city, context_info)
        elif choice == '2':
            question = input("What would you like to know about the forecast? ")
            if question:
                response = predictor.chatbot.chat_about_disaster_prep(question, city, f"Forecast data for {days} days available")
                print(f"\nAnswer: {response}")
        
        return {
            'city': city,
            'forecast': future_predictions,
            'imd_data': imd_data,
            'ai_guidance': guidance
        }
    
    return None

def enhanced_interactive_mode():
    """Enhanced interactive mode with advanced AI chat integration"""
    print("\nENHANCED INTERACTIVE DISASTER PREDICTION SYSTEM")
    print("-" * 60)
    
    predictor = EnhancedTravelRiskPredictor()
    data_fetcher = EnhancedDataFetcher()
    network_checker = NetworkConnectivityChecker()
    
    if not predictor.disaster_models:
        print("No models loaded. Please train models first.")
        return
    
    while True:
        print("\n" + "-" * 60)
        print("ENHANCED ANALYSIS OPTIONS:")
        print("1. Current risk analysis (with IMD data + AI)")
        print("2. 7-day forecast with official warnings + AI guidance") 
        print("3. Chat about disaster preparedness (Advanced NLU)")
        print("4. Explain last prediction with AI insights")
        print("5. What-if scenario analysis")
        print("6. Check different city")
        print("7. Quit")
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == '7':
            print("Stay safe and prepared!")
            break
        
        if choice == '3':
            interactive_disaster_chat_mode()
            continue
        
        if choice in ['1', '2', '4', '5', '6']:
            city = input("Enter city name: ").strip()
            
            if not city:
                print("Please enter a valid city name")
                continue
        
        try:
            if choice == '1':
                # Current analysis with advanced AI
                print(f"Enhanced AI analysis for {city}...")
                
                # Check network
                print("Checking connectivity...")
                network_status = network_checker.check_network_quality()
                
                # Get location data
                print("Fetching comprehensive data...")
                location_data = data_fetcher.get_comprehensive_location_data(city)
                
                if location_data:
                    result = predictor.predict_with_imd_integration(location_data, city)
                    
                    # Show network status
                    network_text = 'GOOD' if network_status['network_available'] else 'POOR'
                    print(f"\nNetwork Status: {network_text} ({network_status['quality']})")
                else:
                    print(f"Could not fetch data for {city}")
            
            elif choice == '2':
                # Enhanced 7-day forecast
                city = input("Enter city name: ").strip()
                if city:
                    enhanced_future_forecast_with_ai_guidance(city, 7)
                else:
                    print("Please enter a valid city name")
            
            elif choice == '4':
                # AI-powered explanation
                location_data = data_fetcher.get_comprehensive_location_data(city)
                if location_data:
                    predictions = predictor.predict_all_disasters(location_data)
                    if predictions:
                        print("\nAI-POWERED PREDICTION EXPLANATION:")
                        print("-" * 40)
                        explanation = predictor.chatbot.explain_prediction(predictions, city, location_data)
                        print(explanation)
                    else:
                        print("No recent predictions to explain")
                else:
                    print(f"Could not fetch data for {city}")
            
            elif choice == '5':
                # What-if scenario analysis
                location_data = data_fetcher.get_comprehensive_location_data(city)
                if location_data:
                    what_if_query = input(f"Ask your what-if question about {city}: ").strip()
                    if what_if_query:
                        response = predictor.handle_what_if_scenario(location_data, city, what_if_query)
                        print(f"\nWhat-if Analysis:\n{response}")
                    else:
                        print("Please enter a valid what-if question.")
                else:
                    print(f"Could not fetch data for {city}")
            
            elif choice == '6':
                continue  # Will ask for city again in next loop
                
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            continue
        except Exception as e:
            print(f"Error: {e}")
            continue

def batch_city_prediction_with_enhanced_ai(cities_list):
    """Enhanced batch prediction with IMD integration and comprehensive AI analysis"""
    print(f"\nENHANCED BATCH ANALYSIS FOR {len(cities_list)} CITIES")
    print("="*60)
    
    predictor = EnhancedTravelRiskPredictor()
    data_fetcher = EnhancedDataFetcher()
    
    if not predictor.disaster_models:
        print("No models loaded. Please train models first.")
        return
    
    results_summary = []
    imd_summary = {}
    
    for i, city in enumerate(cities_list, 1):
        print(f"\n[{i}/{len(cities_list)}] Analyzing {city}...")
        
        # Get location data
        location_data = data_fetcher.get_comprehensive_location_data(city)
        
        if not location_data:
            print(f"Could not fetch data for {city}")
            continue
        
        # Get IMD data
        imd_data = predictor.imd_fetcher.get_comprehensive_imd_data(city)
        
        # Predict disasters
        predictions = predictor.predict_all_disasters(location_data)
        
        if predictions:
            # Find highest risk
            max_risk = max(predictions.values(), key=lambda x: x['probability'])
            max_disaster = max(predictions.keys(), key=lambda x: predictions[x]['probability'])
            
            # Get IMD alert level
            alert_level = 'green'
            if imd_data.get('risk_indicators'):
                alert_level = imd_data['risk_indicators'].get('overall_alert_level', 'green')
            
            results_summary.append({
                'city': city,
                'highest_risk': max_disaster,
                'risk_level': max_risk['risk_level'],
                'probability': max_risk['probability'],
                'imd_alert': alert_level,
                'predictions': predictions,
                'imd_data': imd_data
            })
            
            imd_summary[city] = alert_level
            
            risk_indicator = "HIGH RISK" if max_risk['risk_level'] == 'High' else "MEDIUM" if max_risk['risk_level'] == 'Medium' else "LOW"
            alert_indicator = "RED" if alert_level == 'red' else "ORANGE" if alert_level == 'orange' else "YELLOW" if alert_level == 'yellow' else "GREEN"
            
            print(f"   Risk: {max_disaster.capitalize()} ({risk_indicator} - {max_risk['probability']:.1%})")
            print(f"   IMD Alert: {alert_indicator}")
    
    # Display enhanced summary
    print(f"\n{'='*60}")
    print(f"ENHANCED BATCH ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"{'CITY':<15} | {'TOP RISK':<10} | {'LEVEL':<6} | {'PROB':<6} | {'IMD':<6}")
    print("-" * 60)
    
    for result in results_summary:
        risk_level = {'Low': 'LOW', 'Medium': 'MED', 'High': 'HIGH'}.get(result['risk_level'], 'UNK')
        imd_level = result['imd_alert'].upper()
        print(f"{result['city']:<15} | {result['highest_risk'].capitalize():<10} | {risk_level:<6} | {result['probability']:.1%:<6} | {imd_level:<6}")
    
    # Get comprehensive AI summary
    if results_summary:
        print(f"\nENHANCED AI BATCH ANALYSIS")
        print("="*50)
        
        # Create enhanced context
        high_risk_cities = [r for r in results_summary if r['risk_level'] == 'High']
        red_alert_cities = [r for r in results_summary if r['imd_alert'] == 'red']
        
        batch_context = f"Analyzed {len(results_summary)} cities. "
        if high_risk_cities:
            batch_context += f"HIGH RISK cities: {', '.join([c['city'] for c in high_risk_cities])}. "
        if red_alert_cities:
            batch_context += f"IMD RED ALERT cities: {', '.join([c['city'] for c in red_alert_cities])}. "
        
        guidance = predictor.ai_guidance.get_intelligent_guidance("Multiple Cities", {'batch_summary': batch_context})
        print(guidance)
        print("="*50)
        
        # Offer detailed analysis
        print("\nFOLLOW-UP OPTIONS:")
        print("1. Detailed analysis for specific city")
        print("2. Chat about batch results")
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            detail_choice = input("Enter city name for detailed analysis: ").strip()
            for result in results_summary:
                if detail_choice.lower() in result['city'].lower():
                    print(f"\nDETAILED ANALYSIS FOR {result['city'].upper()}")
                    print("-" * 40)
                    predictor._display_prediction_results(result['predictions'], {})
                    
                    if result['imd_data'].get('risk_indicators'):
                        indicators = result['imd_data']['risk_indicators']
                        high_warnings = indicators.get('high_risk_warnings', [])
                        if high_warnings:
                            print("\nOFFICIAL IMD WARNINGS:")
                            for warning in high_warnings:
                                print(f"  • {warning.get('warning_type', 'Weather Warning')}")
                    break
        elif choice == '2':
            context_info = f"Batch analysis results for {len(results_summary)} cities available"
            predictor.chatbot.interactive_chat_mode(None, context_info)
    
    return results_summary

def train_all_models():
    """Train all disaster prediction models with optional AI enhancement"""
    trainer = MultiDisasterModelTrainer()
    models = trainer.train_disaster_models()
    
    if models:
        print("All models trained successfully!")
        return True
    else:
        print("Model training failed!")
        return False

# Enhanced main execution
if __name__ == "__main__":
    print("ENHANCED AI DISASTER PREDICTION SYSTEM WITH ADVANCED FEATURES")
    print("="*70)
    print("Advanced disaster prediction with:")
    print("• AI-Powered Synthetic Data Generation")
    print("• Advanced Natural Language Understanding")
    print("• What-If Scenario Simulation")
    print("• Automated Alert Summarization & Translation")
    print("• Dynamic Risk Explanation")
    print("• IMD Integration + AI Guidance")
    print("="*70)
    
    # Check if models exist
    import os
    disaster_types = ['flood', 'earthquake', 'landslide', 'cyclone', 'drought']
    models_exist = all(os.path.exists(f'{disaster}_model.pkl') for disaster in disaster_types)
    
    if not models_exist:
        print("\nTraining disaster prediction models...")
        print("=" * 40)
        
        success = train_all_models()
        if not success:
            print("Model training failed. Exiting...")
            exit()
        
        print("All models trained successfully!")
    else:
        print("\nFound existing trained models!")
    
    print("\nADVANCED ANALYSIS MODES:")
    print("1. Enhanced city analysis (Weather + IMD + AI guidance)")
    print("2. Future forecast with AI insights (7 days)")
    print("3. Extended forecast with comprehensive AI (14 days)")
    print("4. Advanced interactive mode (full AI integration)")
    print("5. Enhanced batch city analysis (IMD + AI summaries)")
    print("6. Retrain all models")
    print("7. Advanced AI Chat about disaster planning")
    print("8. Exit")
    
    choice = input("\nEnter your choice (1-8): ").strip()
    
    if choice == '1':
        # Enhanced single city analysis
        city = input("Enter city name: ").strip()
        if city:
            predict_disaster_with_enhanced_ai_guidance(city)
        else:
            print("Please enter a valid city name")
            
    elif choice == '2':
        # Enhanced 7-day forecast
        city = input("Enter city name: ").strip()
        if city:
            enhanced_future_forecast_with_ai_guidance(city, 7)
        else:
            print("Please enter a valid city name")
            
    elif choice == '3':
        # Enhanced 14-day forecast
        city = input("Enter city name: ").strip()
        if city:
            enhanced_future_forecast_with_ai_guidance(city, 14)
        else:
            print("Please enter a valid city name")
            
    elif choice == '4':
        # Advanced interactive mode
        enhanced_interactive_mode()
        
    elif choice == '5':
        # Enhanced batch analysis
        cities_input = input("Enter cities (comma-separated): ").strip()
        if cities_input:
            cities_list = [city.strip() for city in cities_input.split(',')]
            batch_city_prediction_with_enhanced_ai(cities_list)
        else:
            # Default cities
            default_cities = ["Mumbai", "Delhi", "Chennai", "Kolkata", "Bangalore", "Jaipur"]
            print(f"Using default cities: {', '.join(default_cities)}")
            batch_city_prediction_with_enhanced_ai(default_cities)
            
    elif choice == '6':
        # Retrain models
        print("\nRETRAINING ALL MODELS...")
        success = train_all_models()
        if success:
            print("All models retrained successfully!")
        else:
            print("Model retraining failed!")
            
    elif choice == '7':
        # Advanced conversational AI mode
        interactive_disaster_chat_mode()
        
    elif choice == '8':
        print("Thank you for using Enhanced AI Disaster Prediction System!")
        print("Stay safe and prepared!")
        
    else:
        print("Invalid choice. Running advanced interactive mode...")
        enhanced_interactive_mode()
