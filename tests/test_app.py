import pytest
from unittest.mock import patch, MagicMock

def test_home(client):
    response = client.get('/')
    assert response.status_code == 200

@patch('app.data_fetcher')
@patch('app.predictor')
@patch('app.mapbox')
def test_analyze_city(mock_mapbox, mock_predictor, mock_data_fetcher, client):
    # Mock data
    mock_data_fetcher.get_comprehensive_location_data.return_value = {
        'Latitude': 19.076,
        'Longitude': 72.877,
        'Temperature_°C': 30
    }

    mock_predictor.predict_all_disasters.return_value = {
        'flood': {'risk': True, 'probability': 0.8, 'risk_level': 'High'}
    }

    mock_predictor.imd_fetcher.get_comprehensive_imd_data.return_value = {}

    mock_mapbox.get_nearby_shelters.return_value = []

    response = client.get('/api/analyze?city=Mumbai')
    assert response.status_code == 200
    data = response.get_json()
    assert data['location']['latitude'] == 19.076
    assert 'flood' in data['predictions']

def test_analyze_city_no_city(client):
    response = client.get('/api/analyze')
    assert response.status_code == 400
    assert response.get_json()['error'] == 'City name required'
