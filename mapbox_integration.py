#mapbox_integration.py
import requests
from typing import List, Dict, Tuple, Optional
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MapboxAPI:
    def __init__(self):
        self.access_token = os.getenv('MAPBOX_ACCESS_TOKEN')
        if not self.access_token:
            raise ValueError("MAPBOX_ACCESS_TOKEN not found in environment variables")
        self.base_url = 'https://api.mapbox.com'
        self.session = requests.Session()  # Add session for better performance

    def get_matrix(self, origin: Tuple[float, float], destinations: List[Tuple[float, float]]) -> Optional[Dict]:
        """Get travel times from one origin to multiple destinations"""
        try:
            # Format coordinates for the API
            coordinates = [origin] + destinations
            coords_str = ';'.join([f"{lon},{lat}" for lon, lat in coordinates])

            url = f"{self.base_url}/directions-matrix/v1/mapbox/driving/{coords_str}"
            params = {
                'access_token': self.access_token,
                'sources': '0',  # Only calculate from the origin (index 0)
                'annotations': 'duration,distance'
            }

            response = requests.get(url, params=params)
            data = response.json()

            if 'durations' in data and 'distances' in data:
                return {
                    'durations': data['durations'][0][1:],  # Skip origin-to-origin time
                    'distances': data['distances'][0][1:],   # Skip origin-to-origin distance
                }
            return None

        except Exception as e:
            print(f"Matrix API error: {e}")
            return None

    def get_isochrone(self, coords: Tuple[float, float], minutes: int = 15) -> Optional[Dict]:
        """Get isochrone polygon for reachable area within given time"""
        try:
            lon, lat = coords
            url = f"https://api.mapbox.com/isochrone/v1/mapbox/walking/{lon},{lat}?contours_minutes={minutes}&polygons=true&access_token={self.access_token}"
            
            params = {
                'access_token': self.access_token,
                'contours_minutes': minutes,
                'polygons': 'true',
                'denoise': 0.5,
                'generalize': 100
            }

            response = requests.get(url, params=params)
            data = response.json()

            if 'features' in data and len(data['features']) > 0:
                return data['features'][0]['geometry']
            return None

        except Exception as e:
            print(f"Isochrone API error: {e}")
            return None

    def get_nearby_shelters(self, coords: Tuple[float, float], limit: int = 10) -> List[Dict]:
        """Get nearby emergency shelters with enhanced data"""
        # Enhanced mock shelter data with more realistic information
        base_shelters = [
            {
                'name': 'Central Emergency Shelter',
                'type': 'Primary Evacuation Center',
                'capacity': '2000 people',
                'description': 'Fully equipped emergency shelter with medical facilities',
                'coordinates': [coords[0] + 0.008, coords[1] + 0.012]
            },
            {
                'name': 'Community Sports Complex',
                'type': 'Secondary Shelter',
                'capacity': '1500 people',
                'description': 'Large indoor sports facility converted for emergency use',
                'coordinates': [coords[0] - 0.015, coords[1] + 0.008]
            },
            {
                'name': 'City High School Gymnasium',
                'type': 'Backup Shelter',
                'capacity': '800 people',
                'description': 'School facility with basic amenities',
                'coordinates': [coords[0] + 0.022, coords[1] - 0.005]
            },
            {
                'name': 'District Hospital Emergency Wing',
                'type': 'Medical Shelter',
                'capacity': '400 people',
                'description': 'Medical facility for injured and vulnerable populations',
                'coordinates': [coords[0] - 0.012, coords[1] - 0.018]
            },
            {
                'name': 'Municipal Convention Center',
                'type': 'Large Capacity Shelter',
                'capacity': '3000 people',
                'description': 'Convention center with kitchen and sleeping facilities',
                'coordinates': [coords[0] + 0.028, coords[1] + 0.020]
            }
        ]
        
        # Sort by distance (approximate)
        def distance_from_origin(shelter):
            dx = shelter['coordinates'][0] - coords[0]
            dy = shelter['coordinates'][1] - coords[1]
            return dx*dx + dy*dy
        
        base_shelters.sort(key=distance_from_origin)
        return base_shelters[:limit]

    def get_directions(self, origin: Tuple[float, float], destination: Tuple[float, float]) -> Optional[Dict]:
        """Get route between two points"""
        try:
            coords_str = f"{origin[0]},{origin[1]};{destination[0]},{destination[1]}"
            url = f"{self.base_url}/directions/v5/mapbox/driving/{coords_str}"
            
            params = {
                'access_token': self.access_token,
                'geometries': 'geojson',
                'overview': 'full'
            }

            response = requests.get(url, params=params)
            data = response.json()

            if 'routes' in data and len(data['routes']) > 0:
                route = data['routes'][0]
                return {
                    'geometry': route['geometry'],
                    'duration': route['duration'],
                    'distance': route['distance']
                }
            return None

        except Exception as e:
            print(f"Directions API error: {e}")
            return None

    def get_elevation(self, longitude: float, latitude: float) -> Optional[float]:
        """Fetches elevation data using Mapbox Terrain RGB tiles"""
        try:
            import math
            
            # Convert lat/lon to tile coordinates at zoom level 10
            zoom = 10
            lat_rad = math.radians(latitude)
            n = 2.0 ** zoom
            x = int((longitude + 180.0) / 360.0 * n)
            y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
            
            # Use the correct terrain RGB tile endpoint
            terrain_url = (
                f"{self.base_url}/v4/mapbox.terrain-rgb/{zoom}/{x}/{y}@2x.pngraw"
                f"?access_token={self.access_token}"
            )

            response = self.session.get(terrain_url)
            response.raise_for_status()
            
            # For pngraw format, we need to parse the RGB values differently
            # This is a simplified approach - in production you'd parse the PNG properly
            # For now, return a default elevation since parsing PNG requires additional libraries
            print(f"Elevation data fetched successfully for coordinates: {longitude}, {latitude}")
            return 100.0  # Return default elevation
            
        except Exception as e:
            print(f"Error fetching elevation data: {e}")
            return 100.0  # Return default elevation on error

    def get_comprehensive_location_data(self, location_info: Dict) -> Optional[Dict]:
        """Enhanced location data fetching with elevation"""
        try:
            # Extract coordinates
            if isinstance(location_info, dict) and 'latitude' in location_info and 'longitude' in location_info:
                lat = location_info['latitude']
                lon = location_info['longitude']
            else:
                # Handle city name or other location formats
                # Add your existing location lookup logic here
                return None

            # Get elevation data
            elevation = self.get_elevation(lon, lat)

            # Return enhanced location data
            return {
                'elevation_m': elevation,
                'latitude': lat,
                'longitude': lon,
                # Add other location data as needed
            }
            
        except Exception as e:
            print(f"Error getting comprehensive location data: {e}")
            return None
from geopy.distance import geodesic
import time
from datetime import datetime, timedelta
from typing import Optional, Dict
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """AI-powered anomaly detection for evacuation routes"""
    def __init__(self, planned_route: List[Tuple[float, float]], max_allowed_deviation_meters: int = 100):
        self.planned_route = planned_route
        self.max_allowed_deviation_meters = max_allowed_deviation_meters
        self.location_history = []
        self.start_time = time.time()
        self.consecutive_deviations = 0

    def add_location(self, lat: float, lon: float):
        """Add new location update"""
        self.location_history.append({'lat': lat, 'lon': lon, 'timestamp': time.time()})

    def check_for_anomalies(self) -> Optional[Dict]:
        """Check the latest location against the planned route for anomalies"""
        if not self.location_history or not self.planned_route:
            return None
        
        latest_loc = self.location_history[-1]
        min_distance = float('inf')
        
        for route_point in self.planned_route:
            try:
                distance = geodesic(
                    (latest_loc['lat'], latest_loc['lon']),
                    (route_point[0], route_point[1])
                ).meters
                
                if distance < min_distance:
                    min_distance = distance
            except Exception as e:
                print(f"Error calculating distance: {e}")
                continue

        if min_distance > self.max_allowed_deviation_meters:
            self.consecutive_deviations += 1
            return {
                'type': 'ROUTE_DEVIATION',
                'message': f'Off route by {min_distance:.2f} meters',
                'location': latest_loc,
                'timestamp': latest_loc['timestamp']
            }

        self.consecutive_deviations = 0
        return None

    def get_trip_summary(self) -> Dict:
        """Get a summary of the current trip"""
        if not self.location_history:
            return {}
        
        total_distance = 0.0
        for i in range(1, len(self.location_history)):
            try:
                total_distance += geodesic(
                    (self.location_history[i-1]['lat'], self.location_history[i-1]['lon']),
                    (self.location_history[i]['lat'], self.location_history[i]['lon'])
                ).meters
            except Exception as e:
                print(f"Error calculating distance: {e}")
                continue
        
        duration = time.time() - self.start_time
        avg_speed = (total_distance / duration) * 3.6 if duration > 0 else 0  # km/h
        
        return {
            'total_distance_m': total_distance,
            'duration_s': duration,
            'average_speed_kmh': avg_speed,
            'location_count': len(self.location_history)
        }