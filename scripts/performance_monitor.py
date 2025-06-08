#!/usr/bin/env python3
"""
Performance monitoring script for the gold price prediction system.

This script monitors various performance metrics and generates reports
for both staging and production environments.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from prometheus_client.parser import text_string_to_metric_families

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor performance metrics for the gold price prediction system."""
    
    def __init__(self, environment: str = "staging"):
        """
        Initialize the performance monitor.
        
        Args:
            environment: Target environment ("staging" or "production")
        """
        self.environment = environment
        self.base_url = self._get_base_url()
        self.metrics = {}
        self.report_dir = Path("reports/performance")
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_base_url(self) -> str:
        """Get the base URL for the target environment."""
        if self.environment == "production":
            return os.getenv("PRODUCTION_API_URL", "https://api.goldprice.prod")
        else:
            return os.getenv("STAGING_API_URL", "http://localhost:8000")
    
    def check_api_health(self) -> Dict:
        """Check API health and response times."""
        logger.info("Checking API health...")
        
        health_metrics = {
            "status": "unknown",
            "response_time_ms": None,
            "endpoints_tested": 0,
            "endpoints_healthy": 0,
            "errors": []
        }
        
        endpoints = [
            ("/health", "GET"),
            ("/api/v1/health", "GET"),
            ("/api/v1/models/info", "GET"),
        ]
        
        for endpoint, method in endpoints:
            try:
                start_time = time.time()
                url = f"{self.base_url}{endpoint}"
                
                if method == "GET":
                    response = requests.get(url, timeout=30)
                else:
                    response = requests.request(method, url, timeout=30)
                
                response_time = (time.time() - start_time) * 1000
                health_metrics["endpoints_tested"] += 1
                
                if response.status_code == 200:
                    health_metrics["endpoints_healthy"] += 1
                    if health_metrics["response_time_ms"] is None:
                        health_metrics["response_time_ms"] = response_time
                    else:
                        health_metrics["response_time_ms"] = (
                            health_metrics["response_time_ms"] + response_time
                        ) / 2
                else:
                    health_metrics["errors"].append(
                        f"{endpoint}: HTTP {response.status_code}"
                    )
                    
            except Exception as e:
                health_metrics["errors"].append(f"{endpoint}: {str(e)}")
                health_metrics["endpoints_tested"] += 1
        
        # Determine overall status
        if health_metrics["endpoints_healthy"] == health_metrics["endpoints_tested"]:
            health_metrics["status"] = "healthy"
        elif health_metrics["endpoints_healthy"] > 0:
            health_metrics["status"] = "degraded"
        else:
            health_metrics["status"] = "unhealthy"
        
        return health_metrics
    
    def test_prediction_performance(self) -> Dict:
        """Test prediction endpoint performance."""
        logger.info("Testing prediction performance...")
        
        perf_metrics = {
            "predictions_tested": 0,
            "successful_predictions": 0,
            "avg_response_time_ms": 0,
            "min_response_time_ms": float('inf'),
            "max_response_time_ms": 0,
            "errors": [],
            "prediction_accuracy": None
        }
        
        # Sample test data
        test_cases = [
            {
                "historical_prices": [1800.5, 1805.2, 1798.7, 1810.1, 1815.3],
                "days_ahead": 1
            },
            {
                "historical_prices": [1820.0, 1825.5, 1818.2, 1830.7, 1835.1],
                "days_ahead": 3
            },
            {
                "historical_prices": [1840.3, 1845.8, 1838.9, 1850.2, 1855.7],
                "days_ahead": 7
            }
        ]
        
        response_times = []
        
        for i, test_case in enumerate(test_cases):
            try:
                start_time = time.time()
                
                response = requests.post(
                    f"{self.base_url}/api/v1/predict",
                    json=test_case,
                    timeout=30
                )
                
                response_time = (time.time() - start_time) * 1000
                response_times.append(response_time)
                
                perf_metrics["predictions_tested"] += 1
                
                if response.status_code == 200:
                    perf_metrics["successful_predictions"] += 1
                    result = response.json()
                    
                    # Basic validation of response structure
                    if "predicted_price" in result and "confidence" in result:
                        logger.info(f"Test {i+1}: Predicted price = {result['predicted_price']}")
                    else:
                        perf_metrics["errors"].append(
                            f"Test {i+1}: Invalid response structure"
                        )
                else:
                    perf_metrics["errors"].append(
                        f"Test {i+1}: HTTP {response.status_code}"
                    )
                    
            except Exception as e:
                perf_metrics["errors"].append(f"Test {i+1}: {str(e)}")
                perf_metrics["predictions_tested"] += 1
        
        # Calculate response time statistics
        if response_times:
            perf_metrics["avg_response_time_ms"] = np.mean(response_times)
            perf_metrics["min_response_time_ms"] = min(response_times)
            perf_metrics["max_response_time_ms"] = max(response_times)
        
        return perf_metrics
    
    def test_batch_prediction_performance(self) -> Dict:
        """Test batch prediction endpoint performance."""
        logger.info("Testing batch prediction performance...")
        
        batch_metrics = {
            "batch_size": 0,
            "processing_time_ms": 0,
            "throughput_predictions_per_second": 0,
            "memory_usage_mb": None,
            "success": False,
            "errors": []
        }
        
        # Create batch test data
        batch_data = {
            "requests": [
                {
                    "id": f"test_{i}",
                    "historical_prices": [
                        1800 + i * 5 + j for j in range(5)
                    ],
                    "days_ahead": 1
                }
                for i in range(10)  # 10 predictions in batch
            ]
        }
        
        batch_metrics["batch_size"] = len(batch_data["requests"])
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{self.base_url}/api/v1/predict/batch",
                json=batch_data,
                timeout=60
            )
            
            processing_time = (time.time() - start_time) * 1000
            batch_metrics["processing_time_ms"] = processing_time
            
            if response.status_code == 200:
                batch_metrics["success"] = True
                result = response.json()
                
                # Calculate throughput
                if processing_time > 0:
                    batch_metrics["throughput_predictions_per_second"] = (
                        batch_metrics["batch_size"] * 1000 / processing_time
                    )
                
                logger.info(f"Batch processing: {batch_metrics['batch_size']} predictions "
                           f"in {processing_time:.2f}ms")
            else:
                batch_metrics["errors"].append(f"HTTP {response.status_code}")
                
        except Exception as e:
            batch_metrics["errors"].append(str(e))
        
        return batch_metrics
    
    def collect_prometheus_metrics(self) -> Dict:
        """Collect metrics from Prometheus endpoint."""
        logger.info("Collecting Prometheus metrics...")
        
        prometheus_metrics = {
            "available": False,
            "metrics_collected": 0,
            "key_metrics": {},
            "errors": []
        }
        
        try:
            # Try to fetch metrics from Prometheus endpoint
            response = requests.get(
                f"{self.base_url}/metrics",
                timeout=30
            )
            
            if response.status_code == 200:
                prometheus_metrics["available"] = True
                
                # Parse Prometheus metrics
                metrics_text = response.text
                families = text_string_to_metric_families(metrics_text)
                
                key_metric_names = [
                    "http_requests_total",
                    "http_request_duration_seconds",
                    "prediction_requests_total",
                    "prediction_latency_seconds",
                    "model_prediction_accuracy",
                    "system_memory_usage_bytes",
                    "system_cpu_usage_percent"
                ]
                
                for family in families:
                    prometheus_metrics["metrics_collected"] += len(family.samples)
                    
                    if family.name in key_metric_names:
                        prometheus_metrics["key_metrics"][family.name] = {
                            "type": family.type,
                            "help": family.documentation,
                            "samples": [
                                {
                                    "labels": sample.labels,
                                    "value": sample.value
                                }
                                for sample in family.samples
                            ]
                        }
                        
        except Exception as e:
            prometheus_metrics["errors"].append(str(e))
        
        return prometheus_metrics
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        logger.info(f"Generating performance report for {self.environment} environment...")
        
        report = {
            "environment": self.environment,
            "timestamp": datetime.utcnow().isoformat(),
            "monitoring_duration_minutes": 5,  # Standard monitoring window
            "summary": {
                "overall_status": "unknown",
                "performance_score": 0,
                "recommendations": []
            },
            "health_check": {},
            "prediction_performance": {},
            "batch_performance": {},
            "prometheus_metrics": {}
        }
        
        # Run all performance tests
        try:
            report["health_check"] = self.check_api_health()
            report["prediction_performance"] = self.test_prediction_performance()
            report["batch_performance"] = self.test_batch_prediction_performance()
            report["prometheus_metrics"] = self.collect_prometheus_metrics()
            
            # Calculate overall performance score
            score = self._calculate_performance_score(report)
            report["summary"]["performance_score"] = score
            
            # Determine overall status
            if score >= 90:
                report["summary"]["overall_status"] = "excellent"
            elif score >= 75:
                report["summary"]["overall_status"] = "good"
            elif score >= 60:
                report["summary"]["overall_status"] = "fair"
            else:
                report["summary"]["overall_status"] = "poor"
            
            # Generate recommendations
            report["summary"]["recommendations"] = self._generate_recommendations(report)
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            report["summary"]["overall_status"] = "error"
            report["error"] = str(e)
        
        return report
    
    def _calculate_performance_score(self, report: Dict) -> float:
        """Calculate overall performance score (0-100)."""
        score = 0
        max_score = 100
        
        # Health check score (30 points)
        health = report.get("health_check", {})
        if health.get("status") == "healthy":
            score += 30
        elif health.get("status") == "degraded":
            score += 15
        
        # Response time score (25 points)
        response_time = health.get("response_time_ms", float('inf'))
        if response_time < 100:
            score += 25
        elif response_time < 500:
            score += 20
        elif response_time < 1000:
            score += 15
        elif response_time < 2000:
            score += 10
        
        # Prediction performance score (25 points)
        pred_perf = report.get("prediction_performance", {})
        success_rate = 0
        if pred_perf.get("predictions_tested", 0) > 0:
            success_rate = (
                pred_perf.get("successful_predictions", 0) / 
                pred_perf.get("predictions_tested", 1)
            )
        score += success_rate * 25
        
        # Batch performance score (20 points)
        batch_perf = report.get("batch_performance", {})
        if batch_perf.get("success"):
            score += 20
            # Bonus for good throughput
            throughput = batch_perf.get("throughput_predictions_per_second", 0)
            if throughput > 10:
                score += 5  # Bonus points
        
        return min(score, max_score)
    
    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        # Health check recommendations
        health = report.get("health_check", {})
        if health.get("status") != "healthy":
            recommendations.append("Investigate API health issues")
        
        response_time = health.get("response_time_ms", 0)
        if response_time > 1000:
            recommendations.append("Optimize API response times (currently > 1s)")
        elif response_time > 500:
            recommendations.append("Consider optimizing API response times")
        
        # Prediction performance recommendations
        pred_perf = report.get("prediction_performance", {})
        if pred_perf.get("errors"):
            recommendations.append("Fix prediction endpoint errors")
        
        avg_pred_time = pred_perf.get("avg_response_time_ms", 0)
        if avg_pred_time > 2000:
            recommendations.append("Optimize model inference time")
        
        # Batch performance recommendations
        batch_perf = report.get("batch_performance", {})
        if not batch_perf.get("success"):
            recommendations.append("Fix batch prediction endpoint")
        
        throughput = batch_perf.get("throughput_predictions_per_second", 0)
        if throughput < 5:
            recommendations.append("Improve batch processing throughput")
        
        # Prometheus metrics recommendations
        prometheus = report.get("prometheus_metrics", {})
        if not prometheus.get("available"):
            recommendations.append("Enable Prometheus metrics collection")
        
        if not recommendations:
            recommendations.append("Performance is optimal, continue monitoring")
        
        return recommendations
    
    def save_report(self, report: Dict) -> str:
        """Save performance report to file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_report_{self.environment}_{timestamp}.json"
        filepath = self.report_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report saved to {filepath}")
        return str(filepath)
    
    def run_monitoring(self) -> Tuple[Dict, str]:
        """Run complete monitoring and return report and file path."""
        logger.info(f"Starting performance monitoring for {self.environment}")
        
        report = self.generate_performance_report()
        filepath = self.save_report(report)
        
        # Log summary
        summary = report["summary"]
        logger.info(f"Monitoring complete - Status: {summary['overall_status']}, "
                   f"Score: {summary['performance_score']}/100")
        
        return report, filepath


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Monitor performance of gold price prediction system"
    )
    parser.add_argument(
        "--environment",
        choices=["staging", "production"],
        default="staging",
        help="Target environment to monitor"
    )
    parser.add_argument(
        "--output",
        help="Output file path for the report"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        monitor = PerformanceMonitor(args.environment)
        report, filepath = monitor.run_monitoring()
        
        # Print summary to stdout
        summary = report["summary"]
        print(f"Performance Monitoring Summary:")
        print(f"Environment: {args.environment}")
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Performance Score: {summary['performance_score']}/100")
        print(f"Report saved to: {filepath}")
        
        if summary["recommendations"]:
            print("\nRecommendations:")
            for i, rec in enumerate(summary["recommendations"], 1):
                print(f"  {i}. {rec}")
        
        # Exit with appropriate code
        if summary["overall_status"] in ["excellent", "good"]:
            sys.exit(0)
        elif summary["overall_status"] == "fair":
            sys.exit(1)
        else:
            sys.exit(2)
            
    except Exception as e:
        logger.error(f"Performance monitoring failed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
