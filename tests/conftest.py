import pytest
import os
from unittest.mock import patch

# Set dummy env vars before importing app
os.environ['OPENROUTER_API_KEY'] = 'dummy_key'
os.environ['WEATHER_API_KEY'] = 'dummy_key'
os.environ['MAPBOX_ACCESS_TOKEN'] = 'dummy_token'

# Mock dependencies that might be initialized at import time
with patch('ai_engine.AIGuidanceSystem'), \
     patch('ai_engine.DisasterChatbot'), \
     patch('data_fetchers.EnhancedDataFetcher'), \
     patch('server.TravelRiskPredictor'):
    from app import app as flask_app

@pytest.fixture
def app():
    yield flask_app

@pytest.fixture
def client(app):
    return app.test_client()
