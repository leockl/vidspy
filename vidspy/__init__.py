"""
ViDSPy: DSPy-style framework for optimizing text-to-video generation via VBench metric feedback.

Usage:
    pip install vidspy
    
    from vidspy import ViDSPy, VideoSignature, VideoChainOfThought
    
    vidspy = ViDSPy(vlm_backend="openrouter")
    trainset = [Example(prompt="cat jumping", video_path="cat.mp4")]
    
    optimized = vidspy.optimize(
        VideoChainOfThought("prompt -> video"),
        trainset,
        optimizer="mipro_v2"
    )
"""

__version__ = "0.1.1"
__author__ = "ViDSPy Contributors"

# Core classes
from vidspy.core import ViDSPy, Example, VideoExample

# Signatures
from vidspy.signatures import (
    VideoSignature,
    VideoGenerationSignature,
    VideoQualitySignature,
    VideoAlignmentSignature,
)

# Modules
from vidspy.modules import (
    VideoPredict,
    VideoChainOfThought,
    VideoReAct,
    VideoModule,
)

# Optimizers
from vidspy.optimizers import (
    VidBootstrapFewShot,
    VidLabeledFewShot,
    VidMIPROv2,
    VidCOPRO,
    VidGEPA,
)

# Metrics
from vidspy.metrics import (
    VBenchMetric,
    composite_reward,
    quality_score,
    alignment_score,
    CORE_METRICS,
    QUALITY_METRICS,
    ALIGNMENT_METRICS,
)

# Providers
from vidspy.providers import (
    OpenRouterVLM,
    HuggingFaceVLM,
    configure_vlm,
)

# Setup utilities
from vidspy.setup import setup_vbench_models, load_config

__all__ = [
    # Version
    "__version__",
    # Core
    "ViDSPy",
    "Example",
    "VideoExample",
    # Signatures
    "VideoSignature",
    "VideoGenerationSignature",
    "VideoQualitySignature",
    "VideoAlignmentSignature",
    # Modules
    "VideoPredict",
    "VideoChainOfThought",
    "VideoReAct",
    "VideoModule",
    # Optimizers
    "VidBootstrapFewShot",
    "VidLabeledFewShot",
    "VidMIPROv2",
    "VidCOPRO",
    "VidGEPA",
    # Metrics
    "VBenchMetric",
    "composite_reward",
    "quality_score",
    "alignment_score",
    "CORE_METRICS",
    "QUALITY_METRICS",
    "ALIGNMENT_METRICS",
    # Providers
    "OpenRouterVLM",
    "HuggingFaceVLM",
    "configure_vlm",
    # Setup
    "setup_vbench_models",
    "load_config",
]
