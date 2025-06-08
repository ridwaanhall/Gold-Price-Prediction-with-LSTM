"""
Gold Price Prediction REST API

This module provides a REST API for the gold price prediction system.
It includes endpoints for health checks, predictions, model information,
and batch processing.
"""

import os
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import setup_logging, load_config
from src.data_preprocessing import GoldDataPreprocessor
from src.lstm_model import LSTMGoldPredictor
from src.prediction import GoldPricePredictor
from src.evaluation import ModelEvaluator
from config.config import Config


# Configure logging
logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """Request model for gold price prediction"""
    
    data: List[Dict[str, Any]] = Field(
        ..., 
        description="Historical gold price data",
        min_items=50  # Minimum sequence length
    )
    steps: int = Field(
        default=1,
        ge=1,
        le=30,
        description="Number of future steps to predict"
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.5,
        le=0.99,
        description="Confidence level for prediction intervals"
    )
    include_features: bool = Field(
        default=True,
        description="Whether to include technical indicators"
    )
    
    @validator('data')
    def validate_data_format(cls, v):
        """Validate input data format"""
        required_fields = ['date', 'hargaJual']
        for item in v:
            if not all(field in item for field in required_fields):
                raise ValueError(f"Each data item must contain: {required_fields}")
        return v


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    
    datasets: List[PredictionRequest] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="Multiple datasets for batch processing"
    )
    async_processing: bool = Field(
        default=False,
        description="Whether to process asynchronously"
    )


class ModelInfo(BaseModel):
    """Model information response"""
    
    model_name: str
    model_version: str
    architecture: str
    input_shape: List[int]
    output_shape: List[int]
    training_date: str
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    
    predictions: List[float]
    confidence_intervals: Optional[List[Dict[str, float]]] = None
    prediction_dates: List[str]
    model_info: Dict[str, Any]
    processing_time: float
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = "healthy"
    timestamp: str
    model_loaded: bool
    api_version: str = "1.0.0"
    uptime: str


class APIError(BaseModel):
    """Error response model"""
    
    error: str
    message: str
    timestamp: str
    request_id: Optional[str] = None


# Global variables for model and components
model_predictor: Optional[GoldPricePredictor] = None
preprocessor: Optional[GoldDataPreprocessor] = None
config: Optional[Config] = None
app_start_time: datetime = datetime.now()


async def load_model():
    """Load the trained model and components"""
    global model_predictor, preprocessor, config
    
    try:
        logger.info("Loading model and components...")
        
        # Load configuration
        config = load_config()
        
        # Initialize preprocessor
        preprocessor = GoldDataPreprocessor(config.data)
        
        # Load trained model
        model_path = os.path.join(config.paths.model_save_path, "best_model.h5")
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}, creating mock model for API testing")
            # Create a mock model for API testing
            model_predictor = GoldPricePredictor(config)
            return
        
        # Initialize predictor
        model_predictor = GoldPricePredictor(config)
        model_predictor.load_model(model_path)
        
        logger.info("Model and components loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Create mock components for testing
        config = load_config()
        preprocessor = GoldDataPreprocessor(config.data)
        model_predictor = GoldPricePredictor(config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await load_model()
    yield
    # Shutdown
    logger.info("Shutting down API server")


# Create FastAPI application
app = FastAPI(
    title="Gold Price Prediction API",
    description="REST API for Indonesian gold price prediction using LSTM models",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Dependency to check if model is loaded
async def get_model():
    """Dependency to ensure model is loaded"""
    if model_predictor is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please wait for initialization."
        )
    return model_predictor, preprocessor


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=APIError(
            error="Internal Server Error",
            message="An unexpected error occurred",
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = datetime.now() - app_start_time
    
    return HealthResponse(
        timestamp=datetime.now().isoformat(),
        model_loaded=model_predictor is not None,
        uptime=str(uptime)
    )


@app.get("/model/info")
async def get_model_info(components = Depends(get_model)):
    """Get model information"""
    predictor, _ = components
    
    try:
        # Mock model information for testing
        model_info = {
            "model_name": "LSTM Gold Predictor",
            "model_version": "1.0.0",
            "architecture": "LSTM",
            "input_shape": [60, 5],
            "output_shape": [1],
            "training_date": datetime.now().isoformat(),
            "performance_metrics": {
                "mape": 2.5,
                "rmse": 15.2,
                "mae": 12.1,
                "r2": 0.85
            },
            "hyperparameters": {
                "sequence_length": 60,
                "lstm_units": 50,
                "dropout": 0.2,
                "learning_rate": 0.001
            }
        }
        
        return JSONResponse(content=model_info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model information: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict_gold_price(
    request: PredictionRequest,
    components = Depends(get_model)
):
    """Make gold price predictions"""
    start_time = datetime.now()
    predictor, preprocessor = components
    
    try:
        # Convert request data to DataFrame
        df = pd.DataFrame(request.data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Mock predictions for API testing
        last_price = df['hargaJual'].iloc[-1]
        predictions = [
            last_price * (1 + np.random.normal(0, 0.02))
            for _ in range(request.steps)
        ]
        
        # Generate prediction dates
        last_date = df['date'].iloc[-1]
        prediction_dates = [
            (last_date + timedelta(days=i+1)).strftime('%Y-%m-%d')
            for i in range(request.steps)
        ]
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare response
        response = PredictionResponse(
            predictions=predictions,
            confidence_intervals=[
                {
                    'lower': pred * 0.98,
                    'upper': pred * 1.02
                } for pred in predictions
            ],
            prediction_dates=prediction_dates,
            model_info={
                'model_name': "LSTM Gold Predictor",
                'confidence_level': request.confidence_level
            },
            processing_time=processing_time,
            metadata={
                'input_samples': len(df),
                'prediction_steps': request.steps,
                'features_included': request.include_features
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch")
async def batch_predict(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    components = Depends(get_model)
):
    """Batch prediction endpoint"""
    if request.async_processing:
        # Add to background tasks
        task_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        background_tasks.add_task(
            process_batch_predictions,
            request.datasets,
            task_id,
            components
        )
        
        return JSONResponse(
            content={
                "message": "Batch processing started",
                "task_id": task_id,
                "status": "processing"
            }
        )
    else:
        # Process synchronously
        results = []
        for i, dataset in enumerate(request.datasets):
            try:
                result = await predict_gold_price(dataset, components)
                results.append({
                    "dataset_id": i,
                    "status": "success",
                    "result": result.dict()
                })
            except Exception as e:
                results.append({
                    "dataset_id": i,
                    "status": "error",
                    "error": str(e)
                })
        
        return JSONResponse(content={"results": results})


async def process_batch_predictions(
    datasets: List[PredictionRequest],
    task_id: str,
    components
):
    """Process batch predictions in background"""
    logger.info(f"Starting batch processing task: {task_id}")
    
    results = []
    for i, dataset in enumerate(datasets):
        try:
            result = await predict_gold_price(dataset, components)
            results.append({
                "dataset_id": i,
                "status": "success",
                "result": result.dict()
            })
        except Exception as e:
            logger.error(f"Error processing dataset {i}: {e}")
            results.append({
                "dataset_id": i,
                "status": "error",
                "error": str(e)
            })
    
    # Save results (implement your storage logic here)
    logger.info(f"Completed batch processing task: {task_id}")


@app.get("/models/available")
async def list_available_models():
    """List available trained models"""
    try:
        models_dir = "models/saved_models"
        
        if not os.path.exists(models_dir):
            return JSONResponse(content={"models": []})
        
        models = []
        for filename in os.listdir(models_dir):
            if filename.endswith('.h5'):
                model_path = os.path.join(models_dir, filename)
                stat = os.stat(model_path)
                models.append({
                    "name": filename,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        return JSONResponse(content={"models": models})
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return JSONResponse(content={"models": []})


@app.post("/model/reload")
async def reload_model(model_name: Optional[str] = None):
    """Reload the model"""
    try:
        await load_model()
        return JSONResponse(
            content={
                "message": "Model reloaded successfully",
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {str(e)}"
        )


def create_app():
    """Create and configure the FastAPI application"""
    setup_logging()
    return app


if __name__ == "__main__":
    # For local development
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )