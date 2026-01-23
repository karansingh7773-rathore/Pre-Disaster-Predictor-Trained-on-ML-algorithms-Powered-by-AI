import requests
import json
import os
import logging
from datetime import datetime
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
                logger.error(f"IMD District Warnings API returned status code: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error fetching IMD district warnings: {e}")
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
            logger.error(f"Error fetching data for {city_name}: {e}")
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
                logger.error(f"Weather Forecast API Error: {data.get('error', {}).get('message', 'Unknown error')}")
                return None

        except Exception as e:
            logger.error(f"Error fetching forecast data: {e}")
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
                logger.error(f"Weather API Error: {data.get('error', {}).get('message', 'Unknown error')}")
                return None

        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
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
            logger.error(f"Error fetching location data: {e}")
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
                logger.error(f"Weather API Error: {data.get('error', {}).get('message', 'Unknown error')}")
                return None

        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
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
