"""
Tests for ViDSPy core functionality.

These tests verify the basic structure and functionality of ViDSPy
without requiring external services like VBench or OpenRouter.
"""

import pytest
from unittest.mock import MagicMock, patch
import tempfile
import os


class TestImports:
    """Test that all modules can be imported correctly."""
    
    def test_import_vidspy(self):
        """Test main vidspy import."""
        import vidspy
        assert hasattr(vidspy, '__version__')
        assert vidspy.__version__ == "0.1.0"
    
    def test_import_core_classes(self):
        """Test importing core classes."""
        from vidspy import ViDSPy, Example, VideoExample
        assert ViDSPy is not None
        assert Example is not None
        assert VideoExample is not None
    
    def test_import_signatures(self):
        """Test importing signature classes."""
        from vidspy import (
            VideoSignature,
            VideoGenerationSignature,
            VideoQualitySignature,
            VideoAlignmentSignature,
        )
        assert VideoSignature is not None
        assert VideoGenerationSignature is not None
        assert VideoQualitySignature is not None
        assert VideoAlignmentSignature is not None
    
    def test_import_modules(self):
        """Test importing module classes."""
        from vidspy import (
            VideoPredict,
            VideoChainOfThought,
            VideoReAct,
            VideoModule,
        )
        assert VideoPredict is not None
        assert VideoChainOfThought is not None
        assert VideoReAct is not None
        assert VideoModule is not None
    
    def test_import_optimizers(self):
        """Test importing optimizer classes."""
        from vidspy import (
            VidBootstrapFewShot,
            VidLabeledFewShot,
            VidMIPROv2,
            VidCOPRO,
            VidGEPA,
        )
        assert VidBootstrapFewShot is not None
        assert VidLabeledFewShot is not None
        assert VidMIPROv2 is not None
        assert VidCOPRO is not None
        assert VidGEPA is not None
    
    def test_import_metrics(self):
        """Test importing metric functions and constants."""
        from vidspy import (
            VBenchMetric,
            composite_reward,
            quality_score,
            alignment_score,
            CORE_METRICS,
            QUALITY_METRICS,
            ALIGNMENT_METRICS,
        )
        assert VBenchMetric is not None
        assert composite_reward is not None
        assert quality_score is not None
        assert alignment_score is not None
        assert len(CORE_METRICS) == 10
        assert len(QUALITY_METRICS) == 6
        assert len(ALIGNMENT_METRICS) == 4
    
    def test_import_providers(self):
        """Test importing VLM provider classes."""
        from vidspy import (
            OpenRouterVLM,
            HuggingFaceVLM,
            configure_vlm,
        )
        assert OpenRouterVLM is not None
        assert HuggingFaceVLM is not None
        assert configure_vlm is not None


class TestExample:
    """Test Example dataclass."""
    
    def test_create_example(self):
        """Test creating an Example."""
        from vidspy import Example
        
        ex = Example(
            prompt="a cat jumping",
            video_path="/path/to/video.mp4",
        )
        
        assert ex.prompt == "a cat jumping"
        assert ex.video_path == "/path/to/video.mp4"
    
    def test_example_with_metadata(self):
        """Test creating an Example with metadata."""
        from vidspy import Example
        
        ex = Example(
            prompt="a dog running",
            video_path="/path/to/video.mp4",
            metadata={"category": "animals", "duration": 5.0},
        )
        
        assert ex.metadata["category"] == "animals"
        assert ex.metadata["duration"] == 5.0


class TestMetricsConstants:
    """Test metrics constants are correct."""
    
    def test_quality_metrics_list(self):
        """Test QUALITY_METRICS contains expected metrics."""
        from vidspy import QUALITY_METRICS
        
        expected = [
            "subject_consistency",
            "motion_smoothness",
            "temporal_flickering",
            "human_anatomy",
            "aesthetic_quality",
            "imaging_quality",
        ]
        
        assert QUALITY_METRICS == expected
    
    def test_alignment_metrics_list(self):
        """Test ALIGNMENT_METRICS contains expected metrics."""
        from vidspy import ALIGNMENT_METRICS
        
        expected = [
            "object_class",
            "human_action",
            "spatial_relationship",
            "overall_consistency",
        ]
        
        assert ALIGNMENT_METRICS == expected
    
    def test_core_metrics_combined(self):
        """Test CORE_METRICS is combined list."""
        from vidspy import CORE_METRICS, QUALITY_METRICS, ALIGNMENT_METRICS
        
        assert CORE_METRICS == QUALITY_METRICS + ALIGNMENT_METRICS
        assert len(CORE_METRICS) == 10


class TestVBenchMetric:
    """Test VBenchMetric class."""
    
    def test_create_default_metric(self):
        """Test creating VBenchMetric with defaults."""
        from vidspy import VBenchMetric
        
        metric = VBenchMetric()
        
        assert metric.quality_weight == 0.6
        assert metric.alignment_weight == 0.4
    
    def test_create_custom_metric(self):
        """Test creating VBenchMetric with custom weights."""
        from vidspy import VBenchMetric
        
        metric = VBenchMetric(
            quality_weight=0.7,
            alignment_weight=0.3,
            quality_metrics=["motion_smoothness", "temporal_flickering"],
            alignment_metrics=["human_action"],
        )
        
        assert metric.quality_weight == 0.7
        assert metric.alignment_weight == 0.3
        assert "motion_smoothness" in metric.quality_metrics
        assert "human_action" in metric.alignment_metrics


class TestVideoModule:
    """Test video module classes."""

    def test_create_video_predict(self):
        """Test creating VideoPredict module."""
        from vidspy import VideoPredict

        # Just test instantiation without running
        module = VideoPredict("prompt -> video")
        assert module is not None

    def test_create_video_chain_of_thought(self):
        """Test creating VideoChainOfThought module."""
        from vidspy import VideoChainOfThought

        module = VideoChainOfThought("prompt -> video")
        assert module is not None


class TestOptimizers:
    """Test optimizer classes initialization."""
    
    def test_create_bootstrap_optimizer(self):
        """Test creating VidBootstrapFewShot optimizer."""
        from vidspy import VidBootstrapFewShot, composite_reward
        
        optimizer = VidBootstrapFewShot(
            metric=composite_reward,
            max_bootstrapped_demos=4,
        )
        
        assert optimizer is not None
        assert optimizer.max_bootstrapped_demos == 4
    
    def test_create_labeled_optimizer(self):
        """Test creating VidLabeledFewShot optimizer."""
        from vidspy import VidLabeledFewShot, composite_reward
        
        optimizer = VidLabeledFewShot(
            metric=composite_reward,
            k=3,
        )
        
        assert optimizer is not None
        assert optimizer.k == 3
    
    def test_create_mipro_optimizer(self):
        """Test creating VidMIPROv2 optimizer."""
        from vidspy import VidMIPROv2, composite_reward
        
        optimizer = VidMIPROv2(
            metric=composite_reward,
            num_candidates=10,
        )
        
        assert optimizer is not None
        assert optimizer.num_candidates == 10
    
    def test_create_copro_optimizer(self):
        """Test creating VidCOPRO optimizer."""
        from vidspy import VidCOPRO, composite_reward
        
        optimizer = VidCOPRO(
            metric=composite_reward,
            breadth=5,
            depth=3,
        )
        
        assert optimizer is not None
        assert optimizer.breadth == 5
        assert optimizer.depth == 3
    
    def test_create_gepa_optimizer(self):
        """Test creating VidGEPA optimizer."""
        from vidspy import VidGEPA, composite_reward
        
        optimizer = VidGEPA(
            metric=composite_reward,
        )
        
        assert optimizer is not None


class TestViDSPyCore:
    """Test ViDSPy main class."""
    
    def test_create_vidspy_openrouter(self):
        """Test creating ViDSPy with OpenRouter backend."""
        from vidspy import ViDSPy
        
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            vidspy = ViDSPy(vlm_backend="openrouter")
            assert vidspy is not None
            assert vidspy.vlm_backend == "openrouter"
    
    def test_create_vidspy_huggingface(self):
        """Test creating ViDSPy with HuggingFace backend."""
        from vidspy import ViDSPy
        
        vidspy = ViDSPy(
            vlm_backend="huggingface",
            vlm_model="llava-hf/llava-v1.6-mistral-7b-hf",
            device="cpu",
        )
        
        assert vidspy is not None
        assert vidspy.vlm_backend == "huggingface"
    
    def test_available_optimizers(self):
        """Test that ViDSPy lists available optimizers."""
        from vidspy import ViDSPy
        
        expected_optimizers = [
            "bootstrap",
            "labeled",
            "mipro_v2",
            "copro",
            "gepa",
        ]
        
        for opt in expected_optimizers:
            assert opt in ViDSPy.AVAILABLE_OPTIMIZERS


class TestCLI:
    """Test CLI functionality."""
    
    def test_cli_module_exists(self):
        """Test that CLI module can be imported."""
        from vidspy import cli
        assert hasattr(cli, 'main')
    
    def test_cli_has_commands(self):
        """Test that CLI has expected commands."""
        from vidspy.cli import main
        # Just verify the function exists
        assert callable(main)


class TestProviders:
    """Test VLM provider classes."""
    
    def test_openrouter_provider_init(self):
        """Test OpenRouterVLM initialization."""
        from vidspy import OpenRouterVLM
        
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            provider = OpenRouterVLM(model="google/gemini-2.5-flash")
            assert provider is not None
    
    def test_huggingface_provider_init(self):
        """Test HuggingFaceVLM initialization."""
        from vidspy import HuggingFaceVLM
        
        # Just test class exists, actual loading requires GPU
        assert HuggingFaceVLM is not None


class TestSetup:
    """Test setup utilities."""
    
    def test_setup_function_exists(self):
        """Test that setup_vbench_models function exists."""
        from vidspy import setup_vbench_models
        assert callable(setup_vbench_models)


# Integration test markers
@pytest.mark.integration
class TestIntegration:
    """Integration tests (require external services)."""

    @pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY"),
        reason="Requires OPENROUTER_API_KEY environment variable"
    )
    def test_openrouter_inference(self):
        """Test OpenRouter VLM inference with actual API call."""
        from vidspy import OpenRouterVLM

        # Create OpenRouter VLM provider
        vlm = OpenRouterVLM(
            model="google/gemini-2.5-flash",
            api_key=os.environ.get("OPENROUTER_API_KEY")
        )

        # Check provider is available
        assert vlm is not None
        assert vlm.model == "google/gemini-2.5-flash"

        # Test simple text completion (without video)
        response = vlm.complete(
            prompt="What is 2+2? Answer in one word.",
            images=None,
            video_path=None,
            max_tokens=10
        )

        assert response is not None
        assert hasattr(response, 'text')
        assert len(response.text) > 0
        print(f"\n[PASS] OpenRouter API response: {response.text}")

    @pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY"),
        reason="Requires OPENROUTER_API_KEY environment variable"
    )
    def test_vidspy_openrouter_integration(self):
        """Test ViDSPy initialization with OpenRouter backend."""
        from vidspy import ViDSPy, Example

        # Initialize ViDSPy with OpenRouter
        vidspy = ViDSPy(
            vlm_backend="openrouter",
            vlm_model="google/gemini-2.5-flash",
            api_key=os.environ.get("OPENROUTER_API_KEY")
        )

        assert vidspy is not None
        assert vidspy.vlm_backend == "openrouter"

        # Create a simple example (doesn't require actual video file)
        example = Example(
            prompt="a cat jumping",
            video_path="/mock/cat_jump.mp4"
        )

        assert example.prompt == "a cat jumping"
        print(f"\n[PASS] ViDSPy initialized successfully with OpenRouter backend")

    def test_vbench_mock_scoring(self):
        """Test VBench metric evaluation with mock scores (no models needed)."""
        from vidspy.metrics import VBenchMetric, composite_reward, get_vbench
        from vidspy import Example

        # Create VBench metric (works without installation)
        metric = VBenchMetric()

        assert metric.quality_weight == 0.6
        assert metric.alignment_weight == 0.4

        # Test with mock video (doesn't exist, so uses mock scoring)
        example = Example(
            prompt="a person walking through a forest",
            video_path="/nonexistent/mock_video.mp4"  # Doesn't exist -> uses mock
        )

        # Test individual metric scoring via VBenchInterface
        vbench = get_vbench()
        score = vbench.score(
            video_path="/nonexistent/mock_video.mp4",  # Doesn't exist -> mock
            metric="subject_consistency",
            prompt=None
        )
        assert 0.0 <= score <= 1.0
        print(f"\n[PASS] Mock subject_consistency score: {score:.3f}")

        # Test alignment metric with prompt
        alignment_score = vbench.score(
            video_path="/nonexistent/mock_video.mp4",
            metric="human_action",
            prompt="a person walking"
        )
        assert 0.0 <= alignment_score <= 1.0
        print(f"[PASS] Mock human_action score: {alignment_score:.3f}")

        # Test composite reward (combines all metrics)
        prediction = type('obj', (object,), {
            'video_path': "/nonexistent/mock_video.mp4",
            'enhanced_prompt': "enhanced prompt"
        })()

        reward = composite_reward(example, prediction)
        assert 0.0 <= reward <= 1.0
        print(f"[PASS] Mock composite reward: {reward:.3f}")

        # Test VBenchMetric callable
        metric_score = metric(example, prediction)
        assert 0.0 <= metric_score <= 1.0
        print(f"[PASS] VBenchMetric score: {metric_score:.3f}")

        print("[PASS] VBench mock scoring works without model downloads!")

    @pytest.mark.skipif(
        True,  # Skip unless user explicitly wants to test with real VBench
        reason="Requires VBench installation and models (~5-10 GB download)"
    )
    def test_vbench_real_evaluation(self):
        """Test VBench with real models (requires installation)."""
        try:
            import vbench
        except ImportError:
            pytest.skip("VBench not installed")

        from vidspy import setup_vbench_models

        # This would download models
        cache_dir = setup_vbench_models(verbose=True)
        print(f"\n[PASS] VBench models ready at: {cache_dir}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
