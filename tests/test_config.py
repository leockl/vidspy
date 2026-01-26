"""
Tests for configuration loading functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch


class TestConfigLoading:
    """Test YAML configuration loading."""

    def test_load_config_file_not_found(self):
        """Test loading config when file doesn't exist returns empty dict."""
        from vidspy import load_config

        config = load_config()
        assert config == {}

    def test_load_config_from_file(self):
        """Test loading config from a YAML file."""
        from vidspy import load_config

        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
vlm:
  backend: huggingface
  model: llava-hf/llava-v1.6-mistral-7b-hf

cache:
  dir: /tmp/test_cache

hardware:
  device: cpu
  dtype: float32
""")
            config_path = f.name

        try:
            config = load_config(config_path)

            assert config["vlm"]["backend"] == "huggingface"
            assert config["vlm"]["model"] == "llava-hf/llava-v1.6-mistral-7b-hf"
            assert config["cache"]["dir"] == "/tmp/test_cache"
            assert config["hardware"]["device"] == "cpu"
            assert config["hardware"]["dtype"] == "float32"
        finally:
            os.unlink(config_path)

    def test_load_config_invalid_yaml(self):
        """Test loading invalid YAML raises error."""
        from vidspy import load_config

        # Create a temporary file with invalid YAML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid YAML"):
                load_config(config_path)
        finally:
            os.unlink(config_path)

    def test_load_config_missing_file_with_path(self):
        """Test loading non-existent file with explicit path raises error."""
        from vidspy import load_config

        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/to/config.yaml")


class TestViDSPyWithConfig:
    """Test ViDSPy initialization with config file."""

    def test_vidspy_uses_config_defaults(self):
        """Test ViDSPy uses config file values as defaults."""
        from vidspy import ViDSPy

        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
vlm:
  backend: huggingface
  model: test-model

cache:
  dir: /tmp/vidspy_test

hardware:
  device: cpu
""")
            config_path = f.name

        try:
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                vidspy = ViDSPy(config_path=config_path)

                assert vidspy.vlm_backend == "huggingface"
                assert vidspy.vlm_model == "test-model"
                assert vidspy.device == "cpu"
                # Path comparison should be platform-agnostic
                assert vidspy.cache_dir == Path("/tmp/vidspy_test")
        finally:
            os.unlink(config_path)

    def test_vidspy_arguments_override_config(self):
        """Test that arguments passed to ViDSPy override config file."""
        from vidspy import ViDSPy

        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
vlm:
  backend: huggingface
  model: config-model

hardware:
  device: cpu
""")
            config_path = f.name

        try:
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                # Pass arguments that should override config
                vidspy = ViDSPy(
                    vlm_backend="openrouter",
                    vlm_model="override-model",
                    device="cuda",
                    config_path=config_path
                )

                # Verify arguments took precedence
                assert vidspy.vlm_backend == "openrouter"
                assert vidspy.vlm_model == "override-model"
                assert vidspy.device == "cuda"
        finally:
            os.unlink(config_path)

    def test_vidspy_no_config_uses_defaults(self):
        """Test ViDSPy works without config file using built-in defaults."""
        from vidspy import ViDSPy

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            vidspy = ViDSPy()

            # Should use default values
            assert vidspy.vlm_backend == "openrouter"
            assert vidspy.device == "auto"
            assert vidspy.cache_dir == Path.home() / ".cache" / "vidspy"

    def test_vidspy_loads_optimizer_config(self):
        """Test ViDSPy loads optimizer settings from config file."""
        from vidspy import ViDSPy

        # Create a temporary config file with optimizer section
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
vlm:
  backend: openrouter
  model: google/gemini-2.5-flash

optimizer:
  lm: anthropic/claude-3-5-sonnet
  api_key: config-optimizer-key

hardware:
  device: cpu
""")
            config_path = f.name

        try:
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                vidspy = ViDSPy(config_path=config_path)

                # Verify optimizer settings from config
                assert vidspy.optimizer_lm == "anthropic/claude-3-5-sonnet"
                assert vidspy.optimizer_api_key == "config-optimizer-key"
        finally:
            os.unlink(config_path)

    def test_optimizer_args_override_config(self):
        """Test that optimizer arguments override config file."""
        from vidspy import ViDSPy

        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
optimizer:
  lm: openai/gpt-4o-mini
  api_key: config-key
""")
            config_path = f.name

        try:
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                # Pass arguments that should override config
                vidspy = ViDSPy(
                    optimizer_lm="openai/gpt-4o",
                    optimizer_api_key="arg-key",
                    config_path=config_path
                )

                # Verify arguments took precedence
                assert vidspy.optimizer_lm == "openai/gpt-4o"
                assert vidspy.optimizer_api_key == "arg-key"
        finally:
            os.unlink(config_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
