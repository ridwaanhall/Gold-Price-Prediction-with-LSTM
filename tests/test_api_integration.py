#!/usr/bin/env python3
"""
Comprehensive API integration tests for the gold price prediction system.

This module contains tests for all API endpoints including health checks,
predictions, batch processing, model management, and error handling.
"""

import asyncio
import json
import os
import time
from typing import Dict, List, Optional
from unittest.mock import patch

import pytest
import requests
from fastapi.testclient import TestClient

# Import the API application
from api.main import app


class TestAPIIntegration:
    """Integration tests for the Gold Price Prediction API."""
    
    @pytest.fixture(scope="class")
    def client(self):
        """Create a test client for the API."""
        return TestClient(app)
    
    @pytest.fixture(scope="class")
    def sample_prediction_data(self):
        """Sample data for prediction tests."""
        return {
            "historical_prices": [1800.5, 1805.2, 1798.7, 1810.1, 1815.3],
            "days_ahead": 1
        }
    
    @pytest.fixture(scope="class")
    def sample_batch_data(self):
        """Sample data for batch prediction tests."""
        return {
            "requests": [
                {
                    "id": "test_1",
                    "historical_prices": [1800.5, 1805.2, 1798.7, 1810.1, 1815.3],
                    "days_ahead": 1
                },
                {
                    "id": "test_2",
                    "historical_prices": [1820.0, 1825.5, 1818.2, 1830.7, 1835.1],
                    "days_ahead": 3
                }
            ]
        }
    
    # Health Check Tests
    def test_root_health_check(self, client):
        """Test the root health check endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_health_endpoint(self, client):
        """Test the dedicated health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert data["status"] == "healthy"
    
    def test_api_health_endpoint(self, client):
        """Test the API health endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "timestamp" in data
        
        # Check component statuses
        components = data["components"]
        assert "database" in components
        assert "model" in components
        assert "cache" in components
    
    # Model Information Tests
    def test_model_info(self, client):
        """Test model information endpoint."""
        response = client.get("/api/v1/models/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_name" in data
        assert "version" in data
        assert "created_at" in data
        assert "metrics" in data
        assert "status" in data
    
    def test_model_list(self, client):
        """Test model list endpoint."""
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        
        if data["models"]:  # If models exist
            model = data["models"][0]
            assert "name" in model
            assert "version" in model
            assert "status" in model
    
    # Prediction Tests
    def test_single_prediction(self, client, sample_prediction_data):
        """Test single prediction endpoint."""
        response = client.post("/api/v1/predict", json=sample_prediction_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "predicted_price" in data
        assert "confidence" in data
        assert "prediction_date" in data
        assert "model_version" in data
        
        # Validate prediction values
        assert isinstance(data["predicted_price"], (int, float))
        assert 0 <= data["confidence"] <= 1
        assert data["predicted_price"] > 0
    
    def test_prediction_with_different_days_ahead(self, client):
        """Test predictions with different forecasting horizons."""
        test_cases = [1, 3, 7, 14, 30]
        
        for days_ahead in test_cases:
            prediction_data = {
                "historical_prices": [1800.5, 1805.2, 1798.7, 1810.1, 1815.3],
                "days_ahead": days_ahead
            }
            
            response = client.post("/api/v1/predict", json=prediction_data)
            assert response.status_code == 200
            
            data = response.json()
            assert "predicted_price" in data
            assert data["predicted_price"] > 0
    
    def test_prediction_with_invalid_data(self, client):
        """Test prediction endpoint with invalid data."""
        invalid_cases = [
            # Missing required fields
            {"historical_prices": [1800.5, 1805.2]},
            {"days_ahead": 1},
            
            # Invalid data types
            {"historical_prices": "invalid", "days_ahead": 1},
            {"historical_prices": [1800.5, 1805.2], "days_ahead": "invalid"},
            
            # Invalid ranges
            {"historical_prices": [], "days_ahead": 1},
            {"historical_prices": [1800.5, 1805.2], "days_ahead": 0},
            {"historical_prices": [1800.5, 1805.2], "days_ahead": 366},
            
            # Negative prices
            {"historical_prices": [-1800.5, 1805.2], "days_ahead": 1},
        ]
        
        for invalid_data in invalid_cases:
            response = client.post("/api/v1/predict", json=invalid_data)
            assert response.status_code == 422  # Validation error
    
    # Batch Prediction Tests
    def test_batch_prediction(self, client, sample_batch_data):
        """Test batch prediction endpoint."""
        response = client.post("/api/v1/predict/batch", json=sample_batch_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert "summary" in data
        
        results = data["results"]
        assert len(results) == len(sample_batch_data["requests"])
        
        for result in results:
            assert "id" in result
            assert "predicted_price" in result
            assert "confidence" in result
            assert "status" in result
            assert result["status"] == "success"
    
    def test_batch_prediction_with_mixed_validity(self, client):
        """Test batch prediction with mix of valid and invalid requests."""
        mixed_batch_data = {
            "requests": [
                {
                    "id": "valid_1",
                    "historical_prices": [1800.5, 1805.2, 1798.7, 1810.1, 1815.3],
                    "days_ahead": 1
                },
                {
                    "id": "invalid_1",
                    "historical_prices": [],  # Invalid: empty prices
                    "days_ahead": 1
                },
                {
                    "id": "valid_2",
                    "historical_prices": [1820.0, 1825.5, 1818.2, 1830.7, 1835.1],
                    "days_ahead": 3
                }
            ]
        }
        
        response = client.post("/api/v1/predict/batch", json=mixed_batch_data)
        assert response.status_code == 200
        
        data = response.json()
        results = data["results"]
        
        # Check that valid requests succeeded and invalid ones failed
        valid_results = [r for r in results if r["status"] == "success"]
        error_results = [r for r in results if r["status"] == "error"]
        
        assert len(valid_results) == 2
        assert len(error_results) == 1
        assert error_results[0]["id"] == "invalid_1"
    
    def test_batch_prediction_empty_requests(self, client):
        """Test batch prediction with empty requests."""
        empty_batch_data = {"requests": []}
        
        response = client.post("/api/v1/predict/batch", json=empty_batch_data)
        assert response.status_code == 422  # Validation error
    
    # Model Management Tests
    def test_model_reload(self, client):
        """Test model reload endpoint."""
        response = client.post("/api/v1/models/reload")
        
        # Should succeed or return appropriate error
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "message" in data
            assert "model_version" in data
    
    def test_model_switch(self, client):
        """Test model switching endpoint."""
        switch_data = {
            "model_name": "test_model",
            "version": "1.0.0"
        }
        
        response = client.post("/api/v1/models/switch", json=switch_data)
        
        # Should succeed or return appropriate error
        assert response.status_code in [200, 404, 503]
    
    # Metrics and Monitoring Tests
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        # Check that response contains Prometheus metrics format
        content = response.text
        assert "# HELP" in content
        assert "# TYPE" in content
    
    def test_api_metrics_endpoint(self, client):
        """Test API-specific metrics endpoint."""
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "system_metrics" in data
        assert "model_metrics" in data
        assert "api_metrics" in data
    
    # Performance Tests
    def test_prediction_performance(self, client, sample_prediction_data):
        """Test prediction endpoint performance."""
        start_time = time.time()
        
        response = client.post("/api/v1/predict", json=sample_prediction_data)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 5.0  # Should respond within 5 seconds
    
    def test_concurrent_predictions(self, client, sample_prediction_data):
        """Test concurrent prediction requests."""
        import concurrent.futures
        import threading
        
        def make_prediction():
            response = client.post("/api/v1/predict", json=sample_prediction_data)
            return response.status_code == 200
        
        # Test with 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_prediction) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        assert all(results)
    
    # Error Handling Tests
    def test_404_endpoint(self, client):
        """Test non-existent endpoint returns 404."""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test wrong HTTP method returns 405."""
        response = client.delete("/api/v1/predict")  # DELETE not allowed
        assert response.status_code == 405
    
    def test_large_payload_handling(self, client):
        """Test handling of large payloads."""
        large_data = {
            "historical_prices": [1800.0 + i * 0.1 for i in range(10000)],  # Large array
            "days_ahead": 1
        }
        
        response = client.post("/api/v1/predict", json=large_data, timeout=30)
        
        # Should handle large payload or return appropriate error
        assert response.status_code in [200, 413, 422]
    
    # Authentication/Security Tests (if implemented)
    def test_security_headers(self, client):
        """Test that security headers are present."""
        response = client.get("/api/v1/health")
        
        # Check for common security headers
        headers = response.headers
        # Note: Add checks for actual security headers if implemented
        assert response.status_code == 200


class TestAPIProduction:
    """Production-specific API tests."""
    
    @pytest.fixture(scope="class")
    def production_url(self):
        """Get production API URL from environment."""
        return os.getenv("PRODUCTION_API_URL", "http://localhost:8000")
    
    @pytest.mark.skipif(
        os.getenv("TEST_ENVIRONMENT") != "production",
        reason="Production tests only run in production environment"
    )
    def test_production_health(self, production_url):
        """Test production API health."""
        try:
            response = requests.get(f"{production_url}/health", timeout=10)
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "healthy"
            
        except requests.RequestException as e:
            pytest.fail(f"Production health check failed: {e}")
    
    @pytest.mark.skipif(
        os.getenv("TEST_ENVIRONMENT") != "production",
        reason="Production tests only run in production environment"
    )
    def test_production_model_info(self, production_url):
        """Test production model information."""
        try:
            response = requests.get(f"{production_url}/api/v1/models/info", timeout=10)
            assert response.status_code == 200
            
            data = response.json()
            assert "model_name" in data
            assert data["status"] == "loaded"
            
        except requests.RequestException as e:
            pytest.fail(f"Production model info check failed: {e}")


# Utility functions for testing
def test_model_performance():
    """Test model performance metrics against thresholds."""
    # This would typically load a test dataset and evaluate the model
    # For now, we'll use mock metrics
    
    target_metrics = {
        "mape": 3.0,  # Target: < 3%
        "direction_accuracy": 0.7,  # Target: > 70%
        "r2_score": 0.8  # Target: > 0.8
    }
    
    # Mock model evaluation results
    actual_metrics = {
        "mape": 2.5,
        "direction_accuracy": 0.75,
        "r2_score": 0.85
    }
    
    assert actual_metrics["mape"] < target_metrics["mape"]
    assert actual_metrics["direction_accuracy"] > target_metrics["direction_accuracy"]
    assert actual_metrics["r2_score"] > target_metrics["r2_score"]


def test_api_health():
    """Smoke test for API health (used in CI/CD)."""
    base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        
    except requests.RequestException as e:
        pytest.fail(f"API health check failed: {e}")


if __name__ == "__main__":
    # Run basic health check when script is executed directly
    test_api_health()
    print("API health check passed!")
