"""
Shared pytest fixtures and configuration for the test suite.
"""
import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary for tests."""
    return {
        "model_path": "/fake/path/to/model.tflite",
        "confidence_threshold": 0.5,
        "detection_size": (320, 320),
        "iris_size": (64, 64),
        "websocket_url": "ws://localhost:8080",
        "debug": False,
    }


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample image for testing computer vision functions."""
    # Create a simple 640x480 RGB image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some basic patterns for testing
    if CV2_AVAILABLE:
        cv2.rectangle(image, (100, 100), (200, 200), (255, 255, 255), -1)
        cv2.circle(image, (320, 240), 50, (0, 255, 0), -1)
    else:
        # Simple pattern without cv2
        image[100:201, 100:201] = 255  # White rectangle
        # Approximate circle
        y, x = np.ogrid[190:290, 270:370]
        mask = (x - 320)**2 + (y - 240)**2 <= 50**2
        image[190:290, 270:370][mask] = (0, 255, 0)
    return image


@pytest.fixture
def sample_face_landmarks() -> np.ndarray:
    """Provide sample facial landmarks for testing."""
    # 106 2D facial landmarks (common format)
    landmarks = np.random.rand(106, 2) * 300 + 100
    return landmarks.astype(np.float32)


@pytest.fixture
def sample_head_pose() -> Dict[str, np.ndarray]:
    """Provide sample head pose data."""
    return {
        "rotation_vector": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "translation_vector": np.array([0.0, 0.0, -50.0], dtype=np.float32),
        "euler_angles": np.array([15.0, 20.0, 5.0], dtype=np.float32),  # pitch, yaw, roll
    }


@pytest.fixture
def mock_websocket(mocker):
    """Mock websocket client for testing network communication."""
    mock_ws = mocker.MagicMock()
    mock_ws.connected = True
    mock_ws.send.return_value = None
    mock_ws.recv.return_value = '{"status": "ok"}'
    return mock_ws


@pytest.fixture
def mock_tensorflow_model(mocker):
    """Mock TensorFlow Lite model for testing."""
    mock_interpreter = mocker.MagicMock()
    
    # Mock input details
    mock_interpreter.get_input_details.return_value = [{
        'index': 0,
        'shape': [1, 320, 320, 3],
        'dtype': np.float32
    }]
    
    # Mock output details
    mock_interpreter.get_output_details.return_value = [{
        'index': 0,
        'shape': [1, 100, 4],
        'dtype': np.float32
    }]
    
    # Mock inference
    mock_interpreter.invoke.return_value = None
    mock_interpreter.get_tensor.return_value = np.random.rand(1, 100, 4).astype(np.float32)
    
    return mock_interpreter


@pytest.fixture
def capture_stdout(mocker):
    """Capture stdout for testing print statements."""
    import io
    import sys
    
    captured_output = io.StringIO()
    mocker.patch('sys.stdout', captured_output)
    yield captured_output
    sys.stdout = sys.__stdout__


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_camera_capture(mocker, sample_image):
    """Mock OpenCV VideoCapture for testing."""
    mock_cap = mocker.MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, sample_image)
    mock_cap.get.return_value = 30.0  # FPS
    return mock_cap


# Markers for test organization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )