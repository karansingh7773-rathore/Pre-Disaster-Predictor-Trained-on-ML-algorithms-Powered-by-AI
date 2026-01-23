#app.py
from flask import Flask, request, jsonify, send_from_directory
from server import EnhancedTravelRiskPredictor, EnhancedDataFetcher, DisasterChatbot
from mapbox_integration import MapboxAPI
from anomaly_detector import AnomalyDetector
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__, static_url_path='', static_folder='static')
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')

predictor = EnhancedTravelRiskPredictor()
data_fetcher = EnhancedDataFetcher()
mapbox = MapboxAPI()
chatbot = DisasterChatbot()

# Store active trip monitors per session
# In production, you'd use Redis or a proper session store
active_trips = {}

@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'visualization.html')

@app.route('/api/analyze')
def analyze_city():
    city = request.args.get('city')
    if not city:
        return jsonify({'error': 'City name required'}), 400
    
    # Get location data
    location_data = data_fetcher.get_comprehensive_location_data(city)
    if not location_data:
        return jsonify({'error': 'Could not fetch city data'}), 404
    
    # Get predictions
    predictions = predictor.predict_all_disasters(location_data)
    
    # Get IMD warnings
    imd_data = predictor.imd_fetcher.get_comprehensive_imd_data(city)
    
    # Format IMD warnings with polygons
    imd_warnings = []
    if imd_data and 'risk_indicators' in imd_data:
        for warning in imd_data['risk_indicators'].get('high_risk_warnings', []):
            imd_warnings.append({
                'type': warning.get('warning_type', 'Weather Warning'),
                'description': warning.get('description', ''),
                'area': warning.get('area', None)  # GeoJSON polygon if available
            })
    
    # Get nearby shelters
    shelters = mapbox.get_nearby_shelters((location_data['Longitude'], location_data['Latitude']))
    
    return jsonify({
        'location': {
            'latitude': location_data['Latitude'],
            'longitude': location_data['Longitude']
        },
        'predictions': predictions,
        'imd_warnings': imd_warnings,
        'highest_risk': max([p['risk_level'] for p in predictions.values()], default='Low'),
        'shelters': shelters
    })

@app.route('/api/evacuation-route')
def get_evacuation_route():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    
    if not (lat and lon):
        return jsonify({'error': 'Coordinates required'}), 400
    
    try:
        # Get nearby shelters
        shelters = mapbox.get_nearby_shelters((lon, lat))
        if not shelters:
            return jsonify({'error': 'No shelters found nearby'}), 404

        # Find the nearest shelter (first one should be closest)
        nearest_shelter = shelters[0]
        
        # Get route to nearest shelter
        route = mapbox.get_directions((lon, lat), nearest_shelter['coordinates'])
        
        if route and 'geometry' in route:
            return jsonify({
                'route': route,
                'shelter': nearest_shelter
            })
        else:
            return jsonify({'error': 'Could not calculate route to shelter'}), 404
            
    except Exception as e:
        print(f"Error calculating evacuation route: {e}")
        return jsonify({'error': f'Route calculation failed: {str(e)}'}), 500

@app.route('/api/analyze-location')
def analyze_location():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    
    if not (lat and lon):
        return jsonify({'error': 'Coordinates required'}), 400
    
    try:
        # Use the correct method name
        location_data = data_fetcher.get_comprehensive_location_data({
            'latitude': lat,
            'longitude': lon
        })
        if not location_data:
            return jsonify({'error': 'Could not fetch location data'}), 404
            
        # Get predictions
        predictions = predictor.predict_all_disasters(location_data)
        
        return jsonify({
            'location': {'latitude': lat, 'longitude': lon},
            'predictions': predictions,
            'highest_risk': max([p['risk_level'] for p in predictions.values()], default='Low')
        })
        
    except Exception as e:
        print(f"Error analyzing location: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-coords')
def analyze_coordinates():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    
    if not (lat and lon):
        return jsonify({'error': 'Coordinates required'}), 400
    
    # Use the same method as analyze_location
    location_data = data_fetcher.get_comprehensive_location_data({
        'latitude': lat,
        'longitude': lon
    })
    if not location_data:
        return jsonify({'error': 'Could not fetch location data'}), 404
    
    # Get predictions only
    predictions = predictor.predict_all_disasters(location_data)
    
    return jsonify({
        'location': {
            'latitude': lat,
            'longitude': lon
        },
        'predictions': predictions,
        'highest_risk': max([p['risk_level'] for p in predictions.values()], default='Low')
    })

@app.route('/api/isochrone')
def get_isochrone():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    minutes = request.args.get('minutes', type=int, default=15)
    
    if not (lat and lon):
        return jsonify({'error': 'Coordinates required'}), 400
    
    isochrone = mapbox.get_isochrone((lon, lat), minutes)
    return jsonify({'polygon': isochrone})

# NEW ENDPOINTS FOR ANOMALY DETECTION

@app.route('/api/start-trip', methods=['POST'])
def start_trip():
    """Start monitoring an evacuation route"""
    data = request.json
    route_coords = data.get('route')  # Expecting list of [lon, lat] coordinates
    session_id = data.get('session_id', 'default')
    
    if not route_coords:
        return jsonify({'error': 'Route coordinates required'}), 400
    
    # Create new anomaly detector for this trip
    detector = AnomalyDetector(route_coords)
    active_trips[session_id] = detector
    
    return jsonify({
        'status': 'Trip monitoring started',
        'session_id': session_id,
        'route_length': len(route_coords)
    })

@app.route('/api/update-location', methods=['POST'])
def update_location():
    """Update user location and check for anomalies"""
    data = request.json
    lat = data.get('lat')
    lon = data.get('lon')
    session_id = data.get('session_id', 'default')
    
    if not (lat and lon):
        return jsonify({'error': 'Coordinates required'}), 400
    
    if session_id not in active_trips:
        return jsonify({'error': 'No active trip found. Please start monitoring first.'}), 400
    
    detector = active_trips[session_id]
    current_coords = {'lat': lat, 'lon': lon}
    
    # Check for anomalies
    anomalies = detector.check_location(current_coords)
    
    # Get trip summary
    trip_summary = detector.get_trip_summary()
    
    # Check if emergency alert should be triggered
    emergency_alert = detector.should_trigger_emergency_alert(anomalies)
    
    response_data = {
        'status': 'Location updated',
        'anomalies': anomalies,
        'trip_summary': trip_summary,
        'emergency_alert': emergency_alert
    }
    
    # Log anomalies for monitoring
    if anomalies:
        print(f"ANOMALIES DETECTED for session {session_id}: {anomalies}")
        
    if emergency_alert:
        print(f"EMERGENCY ALERT for session {session_id}!")
        # Here you could integrate with emergency services, send SMS, etc.
    
    return jsonify(response_data)

@app.route('/api/stop-trip', methods=['POST'])
def stop_trip():
    """Stop monitoring a trip"""
    data = request.json
    session_id = data.get('session_id', 'default')
    
    if session_id in active_trips:
        # Get final trip summary before removing
        final_summary = active_trips[session_id].get_trip_summary()
        del active_trips[session_id]
        
        return jsonify({
            'status': 'Trip monitoring stopped',
            'final_summary': final_summary
        })
    
    return jsonify({'error': 'No active trip found'}), 404

@app.route('/api/trip-status')
def get_trip_status():
    """Get current status of all active trips"""
    session_id = request.args.get('session_id', 'default')
    
    if session_id in active_trips:
        detector = active_trips[session_id]
        summary = detector.get_trip_summary()
        return jsonify({
            'status': 'active',
            'summary': summary
        })
    
    return jsonify({'status': 'no_active_trip'})

# NEW ENDPOINTS FOR AI CHATBOT

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages with contextual AI responses"""
    data = request.json
    user_question = data.get('question')
    context = data.get('context', {})
    
    if not user_question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        # Get enhanced response from chatbot with context
        ai_response = chatbot.get_contextual_response(user_question, context)
        
        return jsonify({
            'answer': ai_response,
            'timestamp': time.time()
        })
        
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({
            'answer': "I'm sorry, I'm having trouble responding right now. Please try again.",
            'error': True
        })

@app.route('/api/chat/explain-risk', methods=['POST'])
def explain_risk():
    """Explain current risk assessment using AI"""
    data = request.json
    city = data.get('city')
    predictions = data.get('predictions')
    location_data = data.get('location_data', {})
    
    if not (city and predictions):
        return jsonify({'error': 'City and predictions required'}), 400
    
    try:
        explanation = chatbot.explain_prediction(predictions, city, location_data)
        return jsonify({'explanation': explanation})
        
    except Exception as e:
        print(f"Risk explanation error: {e}")
        return jsonify({'error': 'Could not generate explanation'}), 500

@app.route('/api/chat/what-if', methods=['POST'])
def what_if_scenario():
    """Handle what-if scenario analysis"""
    data = request.json
    city = data.get('city')
    scenario_query = data.get('query')
    current_data = data.get('current_data', {})
    
    if not (city and scenario_query):
        return jsonify({'error': 'City and scenario query required'}), 400
    
    try:
        # Use the enhanced predictor's what-if analysis
        analysis = predictor.handle_what_if_scenario(current_data, city, scenario_query)
        return jsonify({'analysis': analysis})
        
    except Exception as e:
        print(f"What-if analysis error: {e}")
        return jsonify({'error': 'Could not perform scenario analysis'}), 500

@app.route('/api/all-evacuation-routes')
def get_all_evacuation_routes():
    """Get routes to all nearby shelters"""
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    
    if not (lat and lon):
        return jsonify({'error': 'Coordinates required'}), 400
    
    try:
        # Get all nearby shelters with details
        shelters = mapbox.get_nearby_shelters((lon, lat))[:5]
        if not shelters:
            return jsonify({'error': 'No shelters found'}), 404

        # Get routes for each shelter with enhanced details
        routes = []
        for shelter in shelters:
            try:
                route_data = mapbox.get_directions((lon, lat), shelter['coordinates'])
                if route_data and 'geometry' in route_data:
                    # Add enhanced shelter info to the response
                    shelter_info = {
                        'name': shelter.get('name', 'Emergency Shelter'),
                        'type': shelter.get('type', 'Shelter'),
                        'capacity': shelter.get('capacity', '1000'),
                        'distance': route_data.get('distance', 0),
                        'duration': route_data.get('duration', 0),
                        'coordinates': shelter['coordinates'],
                        'description': shelter.get('description', 'Emergency evacuation center')
                    }
                    routes.append({
                        'shelter': shelter_info,
                        'geometry': route_data['geometry'],
                        'distance': route_data.get('distance', 0),
                        'duration': route_data.get('duration', 0)
                    })
            except Exception as route_error:
                print(f"Error calculating route to shelter: {route_error}")
                continue

        if not routes:
            return jsonify({'error': 'Could not calculate any routes'}), 404

        return jsonify({'routes': routes})
        
    except Exception as e:
        print(f"Error calculating shelter routes: {e}")
        return jsonify({'error': str(e)}), 500

# NEW ENDPOINTS FOR FORECAST AND BATCH ANALYSIS

@app.route('/api/forecast')
def get_forecast():
    """Get future forecast with AI guidance"""
    city = request.args.get('city')
    days = request.args.get('days', 7, type=int)
    if not city:
        return jsonify({'error': 'City name required'}), 400

    try:
        forecast_data = data_fetcher.get_forecast_data(city, days)
        if not forecast_data:
            return jsonify({'error': 'Could not fetch forecast data'}), 404

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
                        'condition': day_data.get('Weather_Condition', 'N/A')
                    },
                    'risks': daily_predictions
                })
        
        # Ensure AI guidance is properly formatted as a string
        try:
            imd_data = predictor.imd_fetcher.get_comprehensive_imd_data(city)
            guidance = predictor.ai_guidance.get_intelligent_guidance(city, future_predictions, imd_data)
            if not isinstance(guidance, str):
                guidance = str(guidance)  # Convert to string if not already
        except Exception as e:
            print(f"AI guidance error: {e}")
            guidance = "Unable to generate AI guidance at this time."

        return jsonify({
            'city': city,
            'forecast': future_predictions,
            'ai_guidance': guidance
        })
    except Exception as e:
        print(f"Error getting forecast: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    """Perform batch analysis for multiple cities"""
    data = request.json
    cities = data.get('cities')
    if not cities or not isinstance(cities, list):
        return jsonify({'error': 'A list of cities is required'}), 400

    try:
        results_summary = []
        for city in cities:
            location_data = data_fetcher.get_comprehensive_location_data(city)
            if not location_data:
                continue
            
            imd_data = predictor.imd_fetcher.get_comprehensive_imd_data(city)
            predictions = predictor.predict_all_disasters(location_data)
            
            if predictions:
                max_risk = max(predictions.values(), key=lambda x: x['probability'])
                max_disaster = max(predictions.keys(), key=lambda x: predictions[x]['probability'])
                alert_level = 'green'
                if imd_data.get('risk_indicators'):
                    alert_level = imd_data['risk_indicators'].get('overall_alert_level', 'green')
                
                results_summary.append({
                    'city': city,
                    'highest_risk': max_disaster,
                    'risk_level': max_risk['risk_level'],
                    'probability': max_risk['probability'],
                    'imd_alert': alert_level
                })
        
        # AI summary for the batch
        high_risk_cities = [r for r in results_summary if r['risk_level'] == 'High']
        red_alert_cities = [r for r in results_summary if r['imd_alert'] == 'red']
        
        batch_context = f"Analyzed {len(results_summary)} cities. "
        if high_risk_cities:
            batch_context += f"HIGH RISK cities: {', '.join([c['city'] for c in high_risk_cities])}. "
        if red_alert_cities:
            batch_context += f"IMD RED ALERT cities: {', '.join([c['city'] for c in red_alert_cities])}. "
        
        guidance = predictor.ai_guidance.get_intelligent_guidance("Multiple Cities", {'batch_summary': batch_context})

        return jsonify({
            'summary': results_summary,
            'ai_guidance': guidance
        })
    except Exception as e:
        print(f"Error in batch analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/config/mapbox-token')
def get_mapbox_token():
    """Provide Mapbox token to frontend"""
    token = os.getenv('MAPBOX_ACCESS_TOKEN')
    if not token:
        return jsonify({'error': 'Mapbox token not configured'}), 500
    return jsonify({'token': token})

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    port = int(os.getenv('FLASK_PORT', 5000))
    app.run(debug=debug_mode, port=port)