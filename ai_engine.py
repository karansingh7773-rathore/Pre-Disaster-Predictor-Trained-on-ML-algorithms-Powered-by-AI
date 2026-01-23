import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# AI Integration
try:
    from openai import OpenAI
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logger.warning("OpenAI library not found. AI recommendations will be disabled.")

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
                logger.info("Advanced Disaster Preparedness AI Chatbot initialized successfully")
            except Exception as e:
                logger.error(f"Could not initialize chatbot: {e}")
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
            logger.error(f"NLU parsing error: {e}")
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
            logger.error(f"Chatbot error: {e}")
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
            logger.error(f"Explanation error: {e}")
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
                logger.error(f"Error: {e}")
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
            logger.error(f"Chat error: {e}")
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
                logger.info("AI Guidance System initialized successfully")
            except Exception as e:
                logger.error(f"Could not initialize AI system: {e}")
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
            logger.error(f"AI guidance error: {e}")
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
