"""
Pytest configuration and fixtures for ViDSPy tests.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch


# Load .env file if it exists (for integration tests)
# Note: pytest_configure is defined again below to add markers


@pytest.fixture
def mock_video_path():
    """Create a temporary mock video path."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        yield f.name
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def sample_trainset():
    """Create a sample training set."""
    from vidspy import Example
    
    return [
        Example(
            prompt="a cat jumping over a fence",
            video_path="/mock/cat_jump.mp4",
            metadata={"category": "animals"},
        ),
        Example(
            prompt="a person walking in the rain",
            video_path="/mock/walk_rain.mp4",
            metadata={"category": "people"},
        ),
        Example(
            prompt="a car driving on a highway",
            video_path="/mock/car_drive.mp4",
            metadata={"category": "vehicles"},
        ),
    ]


@pytest.fixture
def mock_vbench():
    """Mock VBench module for testing without installation."""
    mock = MagicMock()
    mock.evaluate.return_value = {
        "subject_consistency": 0.85,
        "motion_smoothness": 0.82,
        "temporal_flickering": 0.88,
        "human_anatomy": 0.86,
        "aesthetic_quality": 0.75,
        "imaging_quality": 0.80,
        "object_class": 0.82,
        "human_action": 0.78,
        "spatial_relationship": 0.80,
        "overall_consistency": 0.83,
    }
    return mock


@pytest.fixture
def mock_openrouter_response():
    """Mock OpenRouter API response."""
    return {
        "id": "gen-test123",
        "choices": [
            {
                "message": {
                    "content": "This video shows a cat gracefully jumping over a wooden fence. The motion is smooth and the framing is good.",
                    "role": "assistant",
                }
            }
        ],
    }


@pytest.fixture
def mock_dspy_lm():
    """Mock DSPy language model."""
    mock_lm = MagicMock()
    mock_lm.return_value = "Enhanced prompt for video generation"
    return mock_lm


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    from vidspy.metrics import VBenchInterface
    VBenchInterface._instance = None
    yield


# Markers
def pytest_configure(config):
    """Configure custom pytest markers and load environment variables."""
    # Load .env file if it exists (for integration tests)
    try:
        from dotenv import load_dotenv
        # Look for .env in project root
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass  # python-dotenv not installed

    # Configure markers
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (deselect with '-m \"not integration\"')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU (deselect with '-m \"not gpu\"')"
    )
