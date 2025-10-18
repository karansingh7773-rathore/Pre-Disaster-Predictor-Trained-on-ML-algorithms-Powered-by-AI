#AI_chatbot.py
# Enhanced Multi-Disaster and Travel Risk Prediction System with Advanced AI Integration
import pandas as pd
import numpy as np
import joblib
import requests
import json
from copy import deepcopy
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from typing import Optional, Dict
import warnings
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables at the top
load_dotenv()

# AI Integration
try:
    from openai import OpenAI
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("Warning: OpenAI library not found. AI recommendations will be disabled.")

class IMDDataFetcher:
    """Enhanced data fetcher with Indian Meteorological Department APIs"""
    
    def __init__(self):
        self.base_url = "https://mausam.imd.gov.in/api"
        self.endpoints = {
            'district_warnings': f"{self.base_url}/warnings_district_api.php",
            'nowcast': f"{self.base_url}/nowcastapi.php",
            'rainfall': f"{self.base_url}/statewise_rainfall_api.php",
            'aws_data': "https://city.imd.gov.in/api/aws_data_api.php",
            'basin_qpf': f"{self.base_url}/basin_qpf_api.php",
            'port_warning': f"{self.base_url}/port_wx_api.php",
            'sea_bulletin': f"{self.base_url}/seaarea_bulletin_api.php",
            'coastal_bulletin': f"{self.base_url}/coastal_bulletin_api.php"
        }
        
        # City to state mapping for IMD data
        self.city_state_mapping = {
            'mumbai': 'maharashtra', 'delhi': 'delhi', 'chennai': 'tamil nadu',
            'kolkata': 'west bengal', 'jaipur': 'rajasthan', 'bangalore': 'karnataka',
            'hyderabad': 'telangana', 'pune': 'maharashtra', 'ahmedabad': 'gujarat',
            'surat': 'gujarat', 'lucknow': 'uttar pradesh', 'kanpur': 'uttar pradesh',
            'nagpur': 'maharashtra', 'indore': 'madhya pradesh', 'thane': 'maharashtra',
            'bhopal': 'madhya pradesh', 'visakhapatnam': 'andhra pradesh', 'patna': 'bihar',
            'vadodara': 'gujarat', 'ludhiana': 'punjab', 'agra': 'uttar pradesh',
            'nashik': 'maharashtra', 'faridabad': 'haryana', 'meerut': 'uttar pradesh',
            'rajkot': 'gujarat', 'kalyan': 'maharashtra', 'vasai': 'maharashtra',
            'varanasi': 'uttar pradesh', 'srinagar': 'jammu and kashmir',
            'aurangabad': 'maharashtra', 'dhanbad': 'jharkhand', 'amritsar': 'punjab',
            'allahabad': 'uttar pradesh', 'gwalior': 'madhya pradesh', 'jabalpur': 'madhya pradesh',
            'coimbatore': 'tamil nadu', 'madurai': 'tamil nadu', 'jodhpur': 'rajasthan',
            'kota': 'rajasthan'
        }
    
    def get_district_warnings(self, location):
        """Fetch district-wise weather warnings from IMD"""
        try:
            response = requests.get(self.endpoints['district_warnings'], timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Filter warnings for the specific location/state
                location_warnings = []
                state = self.city_state_mapping.get(location.lower(), location.lower())
                
                if isinstance(data, list):
                    for warning in data:
                        if (warning.get('state', '').lower() == state or 
                            warning.get('district', '').lower() == location.lower()):
                            location_warnings.append(warning)
                
                return location_warnings
            else:
                print(f"IMD District Warnings API returned status code: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error fetching IMD district warnings: {e}")
            return []
    
    def get_comprehensive_imd_data(self, city):
        """Get all available IMD data for a city"""
        imd_data = {
            'district_warnings': self.get_district_warnings(city),
            'fetch_time': datetime.now().isoformat()
        }
        
        # Extract relevant risk indicators from IMD data
        risk_indicators = self._extract_risk_indicators(imd_data, city)
        imd_data['risk_indicators'] = risk_indicators
        
        return imd_data
    
    def _extract_risk_indicators(self, imd_data, city):
        """Extract risk indicators from IMD data"""
        indicators = {
            'high_risk_warnings': [],
            'medium_risk_warnings': [],
            'rainfall_status': 'normal',
            'storm_warnings': [],
            'overall_alert_level': 'green'
        }
        
        # Process district warnings
        warnings = imd_data.get('district_warnings', [])
        for warning in warnings:
            warning_type = warning.get('warning_type', '').lower()
            severity = warning.get('severity', '').lower()
            
            if 'red' in severity or 'extreme' in severity:
                indicators['high_risk_warnings'].append(warning)
                indicators['overall_alert_level'] = 'red'
            elif 'orange' in severity or 'severe' in severity:
                indicators['medium_risk_warnings'].append(warning)
                if indicators['overall_alert_level'] == 'green':
                    indicators['overall_alert_level'] = 'orange'
            elif 'yellow' in severity:
                if indicators['overall_alert_level'] == 'green':
                    indicators['overall_alert_level'] = 'yellow'
        
        return indicators

    def summarize_imd_warning(self, warning_text):
        """Summarizes and translates a technical IMD warning using AI."""
        try:
            ai_client = OpenAI(
                base_url="https://openrouter.ai/api/v1", 
                api_key=os.getenv('OPENROUTER_API_KEY')
            )
            
            prompt = f"""
            The following is a technical weather warning bulletin from the Indian Meteorological Department (IMD).
            Your task is to:
            1. Summarize the key information into 2-3 simple, scannable bullet points.
            2. Translate the summary into Hindi.
            
            Bulletin: "{warning_text}"
            
            Respond in a JSON format with two keys: "summary_en" and "summary_hi".
            """
            
            completion = ai_client.chat.completions.create(
                model="z-ai/glm-4.5-air:free",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            summary = json.loads(completion.choices[0].message.content)
            return summary
        except Exception as e:
            # Fallback to returning the original text on error
            return {"summary_en": warning_text, "summary_hi": ""}

class DisasterChatbot:
    """Advanced Conversational AI for disaster preparedness and Q&A"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        self.client = None
        self.conversation_history = []
        
        if AI_AVAILABLE:
            try:
                self.client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.api_key
                )
                # Test API connection with fast model
                response = self.client.chat.completions.create(
                    model="nvidia/nemotron-nano-9b-v2:free",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant specialized in disaster preparedness and risk assessment."},
                        {"role": "user", "content": "test"}
                    ]
                )
                print("✅ Advanced Disaster Preparedness AI Chatbot initialized successfully")
            except Exception as e:
                print(f"Could not initialize chatbot: {e}")
                self.client = None         
        
        # Knowledge base for disaster preparedness
        self.disaster_knowledge = {
            'flood': {
                'preparation': [
                    "Keep important documents in waterproof containers",
                    "Store emergency supplies on higher floors",
                    "Know your evacuation routes and shelter locations",
                    "Have a battery-powered radio for updates",
                    "Keep sandbags or flood barriers ready"
                ],
                'during': [
                    "Move to higher ground immediately",
                    "Avoid walking or driving through floodwater",
                    "Stay away from electrical equipment if wet",
                    "Listen to emergency broadcasts",
                    "Do not drink floodwater"
                ],
                'after': [
                    "Wait for authorities to declare area safe",
                    "Check for structural damage before entering buildings",
                    "Clean and disinfect everything touched by floodwater",
                    "Take photos for insurance claims",
                    "Boil water until water supply is declared safe"
                ]
            },
            'earthquake': {
                'preparation': [
                    "Secure heavy furniture and appliances",
                    "Identify safe spots in each room (under sturdy tables)",
                    "Keep emergency kit with first aid supplies",
                    "Plan family communication strategy",
                    "Practice drop, cover, and hold on drills"
                ],
                'during': [
                    "Drop to hands and knees immediately",
                    "Take cover under sturdy desk or table",
                    "Hold on and protect your head and neck",
                    "If outdoors, move away from buildings and trees",
                    "If driving, pull over and stop safely"
                ],
                'after': [
                    "Check for injuries and provide first aid",
                    "Inspect home for damage and hazards",
                    "Turn off utilities if damaged",
                    "Stay out of damaged buildings",
                    "Be prepared for aftershocks"
                ]
            }
        }

    def parse_user_intent(self, user_question):
        """Uses a fast AI model to parse user intent and extract entities."""
        if not self.client:
            return {'intent': 'unknown'} # Fallback if AI is down

        prompt = f"""
        Analyze the user's query and classify their intent. Extract key entities.
        User Query: "{user_question}"

        Possible Intents:
        - 'get_safety_tips': User wants to know how to prepare for or act during/after a disaster.
        - 'ask_for_explanation': User is asking 'why' a prediction was made or for more details.
        - 'run_what_if_scenario': User is asking a hypothetical question like "what if rainfall increases?".
        - 'general_question': A general conversational question.

        Respond ONLY with a valid JSON object with the following keys: 'intent', 'disaster_type', 'location', 'timeframe'.
        If a value is not found, use null.
        """
        try:
            completion = self.client.chat.completions.create(
                # Use a fast and free model for real-time interaction
                model="z-ai/glm-4.5-air:free",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"}, # Enforce JSON output
            )
            parsed_intent = json.loads(completion.choices[0].message.content)
            return parsed_intent
        except Exception as e:
            print(f"NLU parsing error: {e}")
            return {'intent': 'unknown'} # Fallback on error
    
    def chat_about_disaster_prep(self, user_question, city=None, context=None):
        """Handle conversational queries about disaster preparedness"""
        if not self.client:
            return self._get_fallback_chat_response(user_question, city)
        
        try:
            # Build context from conversation history and current situation
            system_prompt = """You are a disaster risk analysis expert and emergency advisor specialized in Indian cities, providing advice about disaster preparedness and safety.

            Key Responsibilities:
            1. If user asks about specific city risks:
               - Explain geographical and historical factors that contribute to risk
               - Reference local terrain, weather patterns, and seismic activity
               - Provide location-specific safety recommendations
               - Use historical disaster events as examples when relevant
            
            2. For disaster preparation questions:
               - Give clear, actionable safety guidance
               - Adapt recommendations to local conditions
               - Explain why each measure is important
               - Consider available infrastructure and resources
            
            3. For current risk assessments:
               - Analyze environmental and weather conditions
               - Explain risk levels and contributing factors
               - Provide immediate safety measures if needed
               - Link current conditions to preparation needs

            Remember:
            - Be specific to the city/location when provided
            - Give practical, actionable advice
            - Explain the reasoning behind recommendations
            - Stay focused on user safety and preparedness
            """
            
            # Build location-specific context
            if city:
                system_prompt += f"\n\nCurrent city focus: {city}"
            
            # Add prediction context if available
            if context:
                system_prompt += f"\n\nCurrent risk assessment: {context}"
            
            # Add recent conversation history
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add last few exchanges for context
            for exchange in self.conversation_history[-6:]:  # Last 3 exchanges
                messages.append({"role": "user", "content": exchange['user']})
                messages.append({"role": "assistant", "content": exchange['assistant']})
            
            # Add current question
            messages.append({"role": "user", "content": user_question})
            
            # Create chat completion with model context and conversation history
            completion = self.client.chat.completions.create(
                model="nvidia/nemotron-nano-9b-v2:free",
                messages=messages,
                temperature=0.7,
                max_tokens=500,
                stream=False,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            
            response = completion.choices[0].message.content
            
            # Store in conversation history
            self.conversation_history.append({
                'user': user_question,
                'assistant': response,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only last 10 exchanges
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return response
            
        except Exception as e:
            print(f"Chatbot error: {e}")
            return self._get_fallback_chat_response(user_question)
    
    def explain_prediction(self, prediction_data, city, input_features):
        """Explains why certain predictions were made using the input data."""
        if not self.client:
            return self._get_fallback_explanation(prediction_data)

        try:
            high_risks = [d for d, data in prediction_data.items() if data.get('risk_level') == 'High']
            medium_risks = [d for d, data in prediction_data.items() if data.get('risk_level') == 'Medium']

            # Construct a detailed context with the actual data used for the prediction
            context = f"""
            Here is a disaster risk prediction for {city} and the key data points used by the ML model.
            Explain in a clear, easy-to-understand way *why* these risks were identified, connecting the data to the potential outcome.

            ML Model Input Data:
            - Rainfall: {input_features.get('Rainfall_mm')} mm
            - Temperature: {input_features.get('Temperature_°C')} °C
            - Elevation: {input_features.get('Elevation_m')} meters
            - Proximity to River: {input_features.get('Distance_to_River_km')} km
            - Seismic Activity Level: {input_features.get('Seismic_Activity')}
            - Soil Type: {input_features.get('Soil_Type')}

            Prediction Result:
            - High Risk Disasters: {', '.join(high_risks) if high_risks else 'None'}
            - Medium Risk Disasters: {', '.join(medium_risks) if medium_risks else 'None'}
            
            Focus your explanation on the connection between the input data and the predicted risks.
            """
            
            completion = self.client.chat.completions.create(
                model="deepseek/deepseek-chat-v3.1:free",
                messages=[{"role": "user", "content": context}],
                max_tokens=400,
                temperature=0.5
            )
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"Explanation error: {e}")
            return self._get_fallback_explanation(prediction_data)

    def _get_fallback_chat_response(self, question, city=None):
        """Provide basic responses when AI is unavailable"""
        question_lower = question.lower()
        
        # Detect disaster type
        disaster_type = None
        for disaster in self.disaster_knowledge.keys():
            if disaster in question_lower:
                disaster_type = disaster
                break
                
        if disaster_type:
            # Detect phase (before, during, after)
            if any(word in question_lower for word in ['prepare', 'preparation', 'before', 'kit', 'plan']):
                phase = 'preparation'
            elif any(word in question_lower for word in ['during', 'happening', 'right now', 'emergency']):
                phase = 'during'
            elif any(word in question_lower for word in ['after', 'cleanup', 'recovery', 'damage']):
                phase = 'after'
            else:
                phase = 'preparation'  # default
            
            tips = self.disaster_knowledge[disaster_type].get(phase, [])
            return f"Here are key {phase} tips for {disaster_type}:\n\n" + "\n".join([f"• {tip}" for tip in tips])
        
        return "I can help with disaster preparedness questions! Try asking about emergency planning, specific disasters, or safety measures."
    
    def _get_fallback_explanation(self, prediction_data):
        """Basic explanation when AI is unavailable"""
        explanations = []
        for disaster, data in prediction_data.items():
            risk_level = data.get('risk_level', 'Unknown')
            probability = data.get('probability', 0)
            
            if risk_level in ['High', 'Medium']:
                explanations.append(f"{disaster.upper()}: {risk_level} risk ({probability:.1%} probability)")
        
        if explanations:
            return "Risk Assessment:\n" + "\n".join(explanations)
        else:
            return "Current conditions appear favorable with low risk levels for major disasters."

    def interactive_chat_mode(self, city=None, prediction_context=None):
        """Enhanced interactive chat session with NLU integration"""
        print(f"\n{'='*60}")
        print("🤖 ADVANCED DISASTER PREPAREDNESS AI CHAT")
        print(f"{'='*60}")
        print("Ask me anything about disaster preparation, safety, or emergency planning!")
        print("Type 'quit' to exit, 'clear' to clear conversation history")
        
        if city:
            print(f"Current focus: {city}")
        if prediction_context:
            print("I have current risk assessment data to help answer your questions.")
        
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nStay safe! Remember to keep your emergency kit updated.")
                    break
                
                if user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("Conversation history cleared.")
                    continue
                
                if not user_input:
                    continue
                
                # NEW NLU LOGIC
                parsed_intent = self.parse_user_intent(user_input)
                print("\nDisaster Expert: ", end="")

                intent = parsed_intent.get('intent')
                if intent == 'get_safety_tips':
                    response = self.chat_about_disaster_prep(user_input, city, prediction_context)
                    print(response)
                elif intent == 'ask_for_explanation':
                    if prediction_context:
                        # Create dummy input_features for explanation
                        input_features = {'Rainfall_mm': 10, 'Temperature_°C': 25, 'Elevation_m': 100, 
                                        'Distance_to_River_km': 2, 'Seismic_Activity': 2.0, 'Soil_Type': 'Loam'}
                        explanation = self.explain_prediction(prediction_context, city, input_features)
                        print(explanation)
                    else:
                        print("No recent predictions available to explain. Please run a prediction first.")
                elif intent == 'run_what_if_scenario':
                    print("What-if scenarios require specific prediction data. Please run a city analysis first, then ask your what-if questions.")
                else: # general_question or unknown
                    response = self.chat_about_disaster_prep(user_input, city, prediction_context)
                    print(response)
                
            except KeyboardInterrupt:
                print("\n\nChat ended. Stay safe!")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

    def get_contextual_response(self, user_question: str, context: Optional[Dict] = None) -> str:
        """Alias for chat_about_disaster_prep to maintain API compatibility with frontend"""
        try:
            # Extract city from context if available
            city = context.get('city') if context else None
            
            # Pass both question and context to existing chat method
            response = self.chat_about_disaster_prep(user_question, city, context)
            return response
            
        except Exception as e:
            print(f"Chat error: {e}")
            return "I apologize, but I'm having trouble responding right now. Please try again."
    
class AIGuidanceSystem:
    """Enhanced AI system with explanation capabilities"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        self.client = None
        self.max_tokens = 400
        self.retry_attempts = 3
        
        if AI_AVAILABLE:
            try:
                self.client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.api_key,
                )
                print("✅ AI Guidance System initialized successfully")
            except Exception as e:
                print(f"Could not initialize AI system: {e}")
                self.client = None
    
    def get_intelligent_guidance(self, city, prediction_data, imd_data=None, forecast_data=None):
        """Generate AI-powered guidance with IMD data integration"""
        
        if not self.client:
            return self._get_fallback_guidance(prediction_data)
        
        try:
            # Check if this is a what-if scenario explanation
            if isinstance(prediction_data, str):
                # This is a what-if explanation prompt
                completion = self.client.chat.completions.create(
                    model="nvidia/nemotron-nano-9b-v2:free",
                    messages=[{"role": "user", "content": prediction_data}],
                    max_tokens=self.max_tokens,
                    temperature=0.7
                )
                return completion.choices[0].message.content
            
            # Prepare enhanced context with IMD data
            context = self._prepare_enhanced_context(city, prediction_data, imd_data, forecast_data)
            
            messages = [
                {"role": "system", "content": """You are a disaster risk advisor with access to Indian Meteorological Department data. 
                Provide concise, actionable safety guidance incorporating official weather warnings.
                
                Focus on:
                1. Immediate safety steps based on IMD warnings
                2. Travel and timing recommendations  
                3. Emergency preparedness essentials
                4. Regional context for India
                
                Keep responses brief and prioritized. Use bullet points."""},
                {"role": "user", "content": context}
            ]
            
            completion = self.client.chat.completions.create(
                model="nvidia/nemotron-nano-9b-v2:free",
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0.7
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"AI guidance error: {e}")
            return self._get_fallback_guidance(prediction_data)
    
    def _prepare_enhanced_context(self, city, prediction_data, imd_data, forecast_data):
        """Prepare enhanced context with IMD data"""
        context = f"COMPREHENSIVE RISK ANALYSIS: {city.upper()}\n\n"
        
        # Add IMD official warnings if available
        if imd_data and 'risk_indicators' in imd_data:
            risk_indicators = imd_data['risk_indicators']
            alert_level = risk_indicators.get('overall_alert_level', 'green')
            
            context += f"IMD ALERT LEVEL: {alert_level.upper()}\n"
            
            if risk_indicators.get('high_risk_warnings'):
                context += "OFFICIAL HIGH RISK WARNINGS:\n"
                for warning in risk_indicators['high_risk_warnings']:
                    context += f"- {warning.get('warning_type', 'Weather Warning')}\n"
        
        # Add prediction data
        if isinstance(prediction_data, list):  # Future predictions
            context += "\nFORECAST ANALYSIS:\n"
            for i, pred in enumerate(prediction_data[:3], 1):
                date_str = pred.get('date', f'Day{i}')
                context += f"\n{date_str}:\n"
                
                risks = pred.get('risks', {})
                high_risks = [d for d, r in risks.items() if r.get('risk_level') == 'High']
                if high_risks:
                    context += f"HIGH: {', '.join(high_risks)}\n"
        else:
            context += "\nCURRENT RISK ASSESSMENT:\n"
            risks = prediction_data.get('risks', prediction_data)
            
            high_risks = []
            medium_risks = []
            for disaster_type, risk_data in risks.items():
                if isinstance(risk_data, dict):
                    level = risk_data.get('risk_level', 'Unknown')
                    if level == 'High':
                        high_risks.append(disaster_type)
                    elif level == 'Medium':
                        medium_risks.append(disaster_type)
            
            if high_risks:
                context += f"HIGH RISK: {', '.join(high_risks)}\n"
            if medium_risks:
                context += f"MEDIUM RISK: {', '.join(medium_risks)}\n"
        
        context += "\nProvide specific safety guidance incorporating official warnings."
        return context
    
    def _get_fallback_guidance(self, prediction_data):
        """Enhanced fallback guidance"""
        guidance = ["ENHANCED SAFETY RECOMMENDATIONS:\n"]
        
        # Extract risks
        high_risks = []
        medium_risks = []
        
        if isinstance(prediction_data, list):
            for pred in prediction_data:
                risks = pred.get('risks', {})
                for disaster, data in risks.items():
                    if data.get('risk_level') == 'High' and disaster not in high_risks:
                        high_risks.append(disaster)
                    elif data.get('risk_level') == 'Medium' and disaster not in medium_risks:
                        medium_risks.append(disaster)
        else:
            risks = prediction_data.get('risks', prediction_data)
            for disaster, data in risks.items():
                if isinstance(data, dict):
                    if data.get('risk_level') == 'High':
                        high_risks.append(disaster)
                    elif data.get('risk_level') == 'Medium':
                        medium_risks.append(disaster)
        
        if high_risks:
            guidance.extend([
                "IMMEDIATE PRIORITY ACTIONS:",
                f"• High risk detected: {', '.join(high_risks)}",
                "• Monitor IMD warnings and local news continuously",
                "• Consider postponing non-essential travel",
                "• Prepare emergency supplies and evacuation routes",
                "• Keep important documents in waterproof containers"
            ])
        
        guidance.extend([
            "\nRECOMMENDED ACTIONS:",
            "• Download offline maps and emergency apps",
            "• Keep emergency kit updated (water, food, first aid)",
            "• Maintain charged power banks and battery radio",
            "• Know locations of nearest hospitals and shelters",
            "• Share travel plans with family/friends",
            "• Follow official sources: IMD, NDMA, local administration"
        ])
        
        return "\n".join(guidance)

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
        print(f"🤖 Using AI to generate {n_samples} synthetic data samples for '{disaster_type}'...")
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
            
            print(f"Successfully generated {len(df)} samples with AI.")
            return df

        except Exception as e:
            print(f"AI data generation failed: {e}. Falling back to rule-based generation.")
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
            print(f"Training {disaster_type} prediction model...")
            
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
            
            print(f"{disaster_type.capitalize()} model accuracy: {accuracy:.3f}")
            
            self.models[disaster_type] = model
            joblib.dump(model, f'{disaster_type}_model.pkl')
            df.to_csv(f'{disaster_type}_training_data.csv', index=False)
        
        print("All disaster models trained successfully!")
        return self.models

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
                print(f"Model loaded: {disaster_type.capitalize()}")
            except FileNotFoundError:
                print(f"Model not found: {disaster_type.capitalize()}")
    
    def predict_all_disasters(self, location_data):
        """Predict all disaster risks for a location"""
        if not self.disaster_models:
            print("No models loaded. Please train models first.")
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
            print(f"Error making predictions: {e}")
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
        print("🔮 Simulating 'what-if' scenario...")
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
                            print(f"Warning: Could not convert value for {param}")
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
            print(f"Error in what-if analysis: {str(e)}")
            return self._handle_what_if_fallback(original_data, city, user_query)
            
            # Create AI prompt for analysis
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
            
            # Get AI analysis
            response = self.ai_client.chat.completions.create(
                model="deepseek/deepseek-chat-v3.1:free",
                messages=[
                    {"role": "system", "content": "You are a disaster risk analysis expert. Analyze what-if scenarios and provide structured responses in JSON format."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            # Parse the response
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
                            print(f"Warning: Could not convert value for {param}")
                            continue

            # Get predictions for both scenarios
            original_predictions = self.predict_all_disasters(original_data)
            new_predictions = self.predict_all_disasters(modified_data)

            # Format response
            response = f"\nWHAT-IF ANALYSIS FOR {city.upper()}\n"
            response += "=" * 50 + "\n\n"
            
            # Summary
            response += "SCENARIO SUMMARY:\n"
            response += ai_analysis.get("summary", "Analysis not available") + "\n\n"
            
            # Changes made
            response += "PARAMETER CHANGES:\n"
            for change in ai_analysis.get("changes", []):
                param = change.get("parameter")
                old_val = original_data.get(param, "N/A")
                new_val = modified_data.get(param, "N/A")
                response += f"• {param}: {old_val} → {new_val}\n"
            
            # Risk changes
            response += "\nRISK IMPACT ANALYSIS:\n"
            for disaster in new_predictions.keys():
                old_prob = original_predictions[disaster]['probability']
                new_prob = new_predictions[disaster]['probability']
                change = new_prob - old_prob
                arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
                response += f"• {disaster.capitalize()} Risk: {old_prob:.1%} → {new_prob:.1%} {arrow}\n"

            return response
            
                # Parse the AI response
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
                            print(f"Warning: Could not convert value for {param}")
                            continue

        except Exception as e:
            print(f"Error in what-if analysis: {str(e)}")
            return self._handle_what_if_fallback(original_data, city, user_query)
            
            # Modify the data based on AI analysis
            modified_data = original_data.copy()
            for change in ai_analysis.get('identified_changes', []):
                param = change.get('parameter')
                new_value = change.get('to')
                if param in modified_data and new_value is not None:
                    if isinstance(new_value, (int, float)):
                        modified_data[param] = new_value
                    elif isinstance(new_value, str) and new_value.replace('.', '').isdigit():
                        modified_data[param] = float(new_value)
            
            # Get predictions for both original and modified scenarios
            original_predictions = self.predict_all_disasters(original_data)
            modified_predictions = self.predict_all_disasters(modified_data)
            
            # Generate AI-powered impact analysis
            impact_prompt = f"""
            Compare these two scenarios and explain the changes in disaster risks:
            
            Original Scenario: {original_predictions}
            Modified Scenario: {modified_predictions}
            Changes Made: {ai_analysis['identified_changes']}
            
            Provide a detailed analysis of:
            1. How the changes affect each type of disaster risk
            2. Why certain risks increased or decreased
            3. Specific safety recommendations
            4. Long-term implications
            
            Format as a bulleted list with clear sections.
            """
            
            impact_response = self.ai_client.chat.completions.create(
                model="deepseek/deepseek-chat-v3.1:free",
                messages=[{"role": "user", "content": impact_prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Format the comprehensive response
            response = f"\nWHAT-IF SCENARIO ANALYSIS FOR {city.upper()}\n"
            response += "=" * 50 + "\n\n"
            
            # Scenario Summary
            response += "SCENARIO SUMMARY:\n"
            response += ai_analysis['scenario_summary'] + "\n\n"
            
            # Parameter Changes
            response += "CHANGES ANALYZED:\n"
            for change in ai_analysis['identified_changes']:
                param = change['parameter']
                old_val = original_data.get(param, "N/A")
                new_val = modified_data.get(param, "N/A")
                response += f"• {param}: {old_val} → {new_val}\n"
                response += f"  Reason: {change['reason']}\n"
            
            # Risk Changes
            response += "\nRISK IMPACT ANALYSIS:\n"
            for disaster in modified_predictions.keys():
                old_prob = original_predictions[disaster]['probability']
                new_prob = modified_predictions[disaster]['probability']
                change = new_prob - old_prob
                arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
                response += f"• {disaster.capitalize()} Risk: {old_prob:.1%} → {new_prob:.1%} {arrow}\n"
            
            # AI Impact Analysis
            response += "\nDETAILED IMPACT ANALYSIS:\n"
            response += impact_response.choices[0].message.content + "\n"
            
            # Risk Factors
            response += "\nKEY RISK FACTORS:\n"
            for factor in ai_analysis['risk_factors']:
                response += f"• {factor}\n"
            
            # Safety Recommendations
            response += "\nSAFETY RECOMMENDATIONS:\n"
            for rec in ai_analysis['recommendations']:
                response += f"• {rec}\n"
            
            return response
            
        except Exception as e:
            print(f"Error in what-if analysis: {str(e)}")
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
                
            # Use AI to analyze the scenario
            prompt = f"""
            Analyze this what-if scenario for {city}:
            Current conditions: {original_data}
            User query: {user_query}
            
            Task: Parse this query and identify:
            1. What variables might change
            2. The potential magnitude of changes
            3. Any disaster-specific implications
            4. Required safety adjustments
            
            Format the response as a JSON with:
            - variables_affected: list of affected parameters
            - changes: dictionary of parameter adjustments
            - scenario_analysis: brief analysis
            - safety_recommendations: list of key points
            """
            
            # Get AI analysis
            response = self.ai_client.chat.completions.create(
                model="deepseek/deepseek-chat-v3.1:free",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Parse AI response and modify data
            ai_analysis = json.loads(response.choices[0].message.content)
            
            # Use the AI's suggested changes to modify the scenario
            modified_data = original_data.copy()
            for param, change in ai_analysis['changes'].items():
                if param in modified_data:
                    if isinstance(change, (int, float)):
                        modified_data[param] = change
                    elif isinstance(change, str) and change.replace('.', '').isdigit():
                        modified_data[param] = float(change)
            
            # Run prediction with modified data
            new_predictions = self.predict_all_disasters(modified_data)
            
            # Format response
            response = f"\nWHAT-IF SCENARIO ANALYSIS FOR {city.upper()}\n"
            response += "=" * 50 + "\n\n"
            
            response += "SCENARIO CONTEXT:\n"
            response += ai_analysis['scenario_analysis'] + "\n\n"
            
            response += "VARIABLES AFFECTED:\n"
            for var in ai_analysis['variables_affected']:
                old_val = original_data.get(var, "N/A")
                new_val = modified_data.get(var, "N/A")
                response += f"• {var}: {old_val} → {new_val}\n"
            
            response += "\nRISK CHANGES:\n"
            for disaster, data in new_predictions.items():
                old_prob = self.predict_all_disasters(original_data)[disaster]['probability']
                new_prob = data['probability']
                change = new_prob - old_prob
                direction = "↑" if change > 0 else "↓" if change < 0 else "→"
                response += f"• {disaster.capitalize()}: {old_prob:.1%} → {new_prob:.1%} {direction}\n"
            
            response += "\nSAFETY RECOMMENDATIONS:\n"
            for rec in ai_analysis['safety_recommendations']:
                response += f"• {rec}\n"
            
            return response
                
            # Step 1: Use AI to extract the change from the user's query
            extraction_prompt = f"""
            From the user's query, extract the parameter to change and its new value as a JSON object.
            Example: "what if rainfall in mumbai is 150mm" -> {{"Rainfall_mm": 150}}
            Example: "what if the temperature drops by 5 degrees" -> {{"Temperature_°C": "decrease by 5"}}
            Query: "{user_query}"
            Respond ONLY with the JSON object.
            """
            completion = self.chatbot.client.chat.completions.create(
                model="mistralai/mistral-7b-instruct:free",
                messages=[{"role": "user", "content": extraction_prompt}],
                response_format={"type": "json_object"},
            )
            modifications = json.loads(completion.choices[0].message.content)
            
            # Step 2: Create the modified data
            modified_data = original_data.copy()
            for key, value in modifications.items():
                if "decrease by" in str(value):
                    amount = float(str(value).split()[-1])
                    modified_data[key] -= amount
                elif "increase by" in str(value):
                    amount = float(str(value).split()[-1])
                    modified_data[key] += amount
                else:
                    modified_data[key] = float(value)

            # Step 3: Get original and new predictions
            original_prediction = self.predict_all_disasters(original_data)
            new_prediction = self.predict_all_disasters(modified_data)

            # Step 4: Use AI to generate a comparative explanation
            explanation_prompt = f"""
            A user in {city} asked a "what-if" question: "{user_query}".
            Explain the change in disaster risk in a simple, clear way.

            Original Risk Profile: {json.dumps(original_prediction)}
            New Risk Profile (after change): {json.dumps(new_prediction)}
            
            Start your answer directly. For example: "If the rainfall increased to 150mm, the flood risk would..."
            """
            explanation = self.ai_guidance.get_intelligent_guidance(city, explanation_prompt)
            return explanation

        except Exception as e:
            return f"Sorry, I couldn't process that scenario. Error: {e}"

    def predict_with_imd_integration(self, location_data, city_name):
        """Enhanced prediction with IMD data integration"""
        if not self.disaster_models:
            print("No models loaded. Please train models first.")
            return None
        
        try:
            # Get IMD data
            print("Fetching IMD official data...")
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
            print(f"Error in enhanced prediction: {e}")
            return None

class EnhancedDataFetcher:
    def __init__(self):
        self.weather_api_key = os.getenv('WEATHER_API_KEY')
        if not self.weather_api_key:
            raise ValueError("WEATHER_API_KEY not found in environment variables")
        self.weather_url = "http://api.weatherapi.com/v1/current.json"
        self.forecast_url = "http://api.weatherapi.com/v1/forecast.json"
        from mapbox_integration import MapboxAPI
        self.mapbox = MapboxAPI()
        
    def get_comprehensive_location_data(self, city_name):
        """Get comprehensive data for location analysis"""
        try:
            weather_data = self._get_weather_data(city_name)
            if not weather_data:
                return None
            
            location_data = self._get_location_specific_data(city_name)
            comprehensive_data = {**weather_data, **location_data}
            
            return comprehensive_data
            
        except Exception as e:
            print(f"Error fetching data for {city_name}: {e}")
            return None
    
    def get_forecast_data(self, city_name, days=7):
        """Get weather forecast data"""
        try:
            params = {
                'key': self.weather_api_key,
                'q': city_name,
                'days': min(days, 10),
                'aqi': 'yes',
                'alerts': 'yes'
            }
            
            response = requests.get(self.forecast_url, params=params)
            data = response.json()
            
            if response.status_code == 200:
                forecast_data = []
                location = data['location']
                
                for day in data['forecast']['forecastday']:
                    day_data = day['day']
                    date = day['date']
                    
                    forecast_entry = {
                        'date': date,
                        'Temperature_°C': day_data['avgtemp_c'],
                        'Humidity_': day_data['avghumidity'],
                        'Pressure_hPa': 1013,
                        'Wind_Speed_kmh': day_data['maxwind_kph'],
                        'Rainfall_mm': day_data.get('totalprecip_mm', 0),
                        'Latitude': location['lat'],
                        'Longitude': location['lon'],
                        'Weather_Condition': day_data['condition']['text']
                    }
                    
                    location_data = self._get_location_specific_data(city_name)
                    forecast_entry.update(location_data)
                    
                    forecast_data.append(forecast_entry)
                
                return forecast_data
            else:
                print(f"Weather Forecast API Error: {data.get('error', {}).get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"Error fetching forecast data: {e}")
            return None
    
    def _get_weather_data(self, city_name):
        """Fetch weather data from API"""
        try:
            params = {
                'key': self.weather_api_key,
                'q': city_name,
                'aqi': 'no'
            }
            
            response = requests.get(self.weather_url, params=params)
            data = response.json()
            
            if response.status_code == 200:
                current = data['current']
                location = data['location']
                
                return {
                    'Temperature_°C': current['temp_c'],
                    'Humidity_': current['humidity'],
                    'Pressure_hPa': current['pressure_mb'],
                    'Wind_Speed_kmh': current['wind_kph'],
                    'Rainfall_mm': current.get('precip_mm', 0),
                    'Latitude': location['lat'],
                    'Longitude': location['lon']
                }
            else:
                print(f"Weather API Error: {data.get('error', {}).get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def get_comprehensive_location_data(self, location_info):
        """Get comprehensive data for location analysis"""
        try:
            # If we have direct coordinates
            if isinstance(location_info, dict) and 'latitude' in location_info and 'longitude' in location_info:
                weather_data = self._get_weather_data_from_coords(
                    location_info['latitude'],
                    location_info['longitude']
                )
            else:
                # Treat as city name
                weather_data = self._get_weather_data(location_info)
                
            if not weather_data:
                return None
            
            location_data = self._get_location_specific_data(
                location_info if isinstance(location_info, str) else None,
                weather_data.get('Latitude'),
                weather_data.get('Longitude')
            )
            
            return {**weather_data, **location_data}
            
        except Exception as e:
            print(f"Error fetching location data: {e}")
            return None

    def _get_weather_data_from_coords(self, lat, lon):
        """Fetch weather data using coordinates"""
        try:
            params = {
                'key': self.weather_api_key,
                'q': f"{lat},{lon}",
                'aqi': 'no'
            }
            
            response = requests.get(self.weather_url, params=params)
            data = response.json()
            
            if response.status_code == 200:
                current = data['current']
                location = data['location']
                
                return {
                    'Temperature_°C': current['temp_c'],
                    'Humidity_': current['humidity'],
                    'Pressure_hPa': current['pressure_mb'],
                    'Wind_Speed_kmh': current['wind_kph'],
                    'Rainfall_mm': current.get('precip_mm', 0),
                    'Latitude': location['lat'],
                    'Longitude': location['lon']
                }
            else:
                print(f"Weather API Error: {data.get('error', {}).get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return None

    def _get_location_specific_data(self, city_name=None, lat=None, lon=None):
        """Get location data using either city name or coordinates"""
        # Use Mapbox for elevation data
        elevation = None
        if lat is not None and lon is not None:
            elevation = self.mapbox.get_elevation(lon, lat) or 100.0
        
        return {
            'Population_Density': 5000.0,
            'Historical_Events': 2,
            'Seismic_Activity': 2.0,
            'Slope_Angle': 3.0,
            'Infrastructure': 'Medium',
            'Land_Cover': 'Urban',
            'Soil_Type': 'Loam',
            'Season': self._get_current_season(),
            'Water_Level_m': 2.0,
            'River_Discharge_ms': 100.0,
            'Distance_to_River_km': 5.0,
            'Elevation_m': elevation or 100.0
        }

    def _get_current_season(self):
        """Determine current season based on month"""
        month = datetime.now().month
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8, 9]:
            return 'Monsoon'
        else:
            return 'Summer'

class NetworkConnectivityChecker:
    def __init__(self):
        self.connectivity_apis = [
            "https://www.google.com",
            "https://www.cloudflare.com", 
            "https://1.1.1.1"
        ]
    
    def check_network_quality(self, location=None):
        """Check network connectivity and quality"""
        try:
            import time
            
            results = {
                'network_available': False,
                'response_times': [],
                'quality': 'Unknown'
            }
            
            for api in self.connectivity_apis:
                try:
                    start_time = time.time()
                    response = requests.get(api, timeout=5)
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        response_time = (end_time - start_time) * 1000
                        results['response_times'].append(response_time)
                        results['network_available'] = True
                except:
                    continue
            
            if results['response_times']:
                avg_response = sum(results['response_times']) / len(results['response_times'])
                
                if avg_response < 100:
                    results['quality'] = 'Excellent'
                elif avg_response < 300:
                    results['quality'] = 'Good'
                elif avg_response < 1000:
                    results['quality'] = 'Fair'
                else:
                    results['quality'] = 'Poor'
                    

                results['avg_response_time'] = round(avg_response, 2)
            
            return results
            
        except Exception as e:
            return {
                'network_available': False,
                'quality': 'Error',
                'error': str(e)
            }

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