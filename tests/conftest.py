"""
Test configuration for pytest.
"""

import pytest
import warnings
import tensorflow as tf
import os

# Suppress TensorFlow warnings for cleaner test output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


@pytest.fixture(scope="session", autouse=True)
def setup_tensorflow():
    """Setup TensorFlow for testing."""
    # Set memory growth for GPU (if available)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)


@pytest.fixture(autouse=True)
def reset_tf_session():
    """Reset TensorFlow session before each test."""
    tf.keras.backend.clear_session()


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark slow tests
        if any(keyword in item.name.lower() for keyword in ['train', 'optimization', 'cross_validation']):
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if 'integration' in item.name.lower() or 'pipeline' in item.name.lower():
            item.add_marker(pytest.mark.integration)
        
        # Mark GPU tests
        if any(keyword in item.name.lower() for keyword in ['gpu', 'cuda']):
            item.add_marker(pytest.mark.gpu)
