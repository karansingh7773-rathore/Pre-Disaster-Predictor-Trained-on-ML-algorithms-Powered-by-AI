import pytest
from unittest.mock import patch, MagicMock
from data_fetchers import IMDDataFetcher

@pytest.fixture
def imd_fetcher():
    return IMDDataFetcher()

@patch('data_fetchers.requests.get')
def test_get_district_warnings(mock_get, imd_fetcher):
    # Mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {'state': 'Maharashtra', 'district': 'Mumbai', 'warning_type': 'Heavy Rain'}
    ]
    mock_get.return_value = mock_response

    warnings = imd_fetcher.get_district_warnings('Mumbai')
    assert len(warnings) == 1
    assert warnings[0]['warning_type'] == 'Heavy Rain'

@patch('data_fetchers.requests.get')
def test_get_district_warnings_error(mock_get, imd_fetcher):
    mock_get.side_effect = Exception("API Error")
    warnings = imd_fetcher.get_district_warnings('Mumbai')
    assert warnings == []
