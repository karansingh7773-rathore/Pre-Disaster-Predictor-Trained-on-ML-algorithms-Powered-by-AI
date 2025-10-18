# anomaly_detector.py
import time
from geopy.distance import geodesic
from datetime import datetime
import json

class AnomalyDetector:
    """AI-powered anomaly detection for evacuation routes"""
    
    def __init__(self, planned_route_coords):
        self.planned_route = planned_route_coords
        self.last_update_time = time.time()
        self.start_time = time.time()
        self.location_history = []
        
        # Detection thresholds
        self.max_allowed_deviation_meters = 150
        self.max_inactivity_seconds = 120
        self.max_stationary_seconds = 300
        self.min_speed_kmh = 2
        self.max_speed_kmh = 80
        
        # State tracking
        self.consecutive_deviations = 0
        self.last_position = None
        self.stationary_start_time = None
        
    def check_location(self, current_coords, timestamp=None):
        """Analyze current location for anomalies"""
        if timestamp is None:
            timestamp = time.time()
            
        anomalies = []
        
        # Store location in history
        location_entry = {
            'coords': current_coords,
            'timestamp': timestamp,
            'lat': current_coords['lat'],
            'lon': current_coords['lon']
        }
        self.location_history.append(location_entry)
        
        # Keep only last 20 locations
        if len(self.location_history) > 20:
            self.location_history = self.location_history[-20:]
        
        # Check for route deviation
        deviation_info = self._check_route_deviation(current_coords)
        if deviation_info:
            anomalies.append(deviation_info)
        
        # Update state
        self.last_update_time = timestamp
        self.last_position = current_coords
        
        return anomalies
    
    def _check_route_deviation(self, current_coords):
        """Check if user has deviated from planned route"""
        if not self.planned_route:
            return None
            
        # Find minimum distance to any point on the planned route
        min_distance = float('inf')
        
        for route_point in self.planned_route:
            try:
                # route_point is [lon, lat], current_coords is {'lat': x, 'lon': y}
                distance = geodesic(
                    (current_coords['lat'], current_coords['lon']),
                    (route_point[1], route_point[0])  # Convert to (lat, lon)
                ).meters
                
                if distance < min_distance:
                    min_distance = distance
            except Exception as e:
                print(f"Error calculating distance: {e}")
                continue
        
        if min_distance > self.max_allowed_deviation_meters:
            self.consecutive_deviations += 1
            
            severity = 'HIGH' if min_distance > 500 else 'MEDIUM'
                
            return {
                'type': 'ROUTE_DEVIATION',
                'severity': severity,
                'message': f'Off planned route by {int(min_distance)}m',
                'data': {
                    'deviation_meters': int(min_distance),
                    'consecutive_count': self.consecutive_deviations
                }
            }
        else:
            self.consecutive_deviations = 0
            
        return None
    
    def get_trip_summary(self):
        """Get summary of the current trip"""
        if not self.location_history:
            return {
                'total_distance_km': 0,
                'elapsed_time_minutes': 0,
                'average_speed_kmh': 0,
                'location_updates': 0
            }
            
        total_distance = 0
        for i in range(1, len(self.location_history)):
            try:
                prev_loc = self.location_history[i-1]
                curr_loc = self.location_history[i]
                
                distance = geodesic(
                    (prev_loc['lat'], prev_loc['lon']),
                    (curr_loc['lat'], curr_loc['lon'])
                ).meters
                total_distance += distance
            except Exception as e:
                print(f"Error calculating trip distance: {e}")
                continue
        
        elapsed_time = time.time() - self.start_time
        
        return {
            'total_distance_km': round(total_distance / 1000, 2),
            'elapsed_time_minutes': round(elapsed_time / 60, 1),
            'average_speed_kmh': round((total_distance / 1000) / (elapsed_time / 3600), 1) if elapsed_time > 0 else 0,
            'location_updates': len(self.location_history)
        }
    
    def should_trigger_emergency_alert(self, anomalies):
        """Determine if anomalies warrant an emergency alert"""
        high_severity_count = sum(1 for a in anomalies if a.get('severity') == 'HIGH')
        return high_severity_count > 0