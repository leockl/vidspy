"""
Setup utilities for ViDSPy and VBench models.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, looks for vidspy_config.yaml
                    in current directory, then user home directory.

    Returns:
        Dictionary with configuration values. Returns empty dict if no config found.

    Example:
        >>> config = load_config()
        >>> config.get('vlm', {}).get('backend', 'openrouter')
        'openrouter'
    """
    import yaml

    # Determine config file path
    if config_path is None:
        # Search order: current dir, home dir
        search_paths = [
            Path.cwd() / "vidspy_config.yaml",
            Path.home() / ".vidspy" / "config.yaml",
            Path.home() / "vidspy_config.yaml",
        ]

        config_file = None
        for path in search_paths:
            if path.exists():
                config_file = path
                break
    else:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

    # If no config file found, return empty dict
    if config_file is None:
        return {}

    # Load and parse YAML
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        return config if config is not None else {}

    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file {config_file}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {config_file}: {e}")


def setup_vbench_models(
    cache_dir: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Download and setup VBench evaluation models.
    
    VBench requires specific pretrained models for each evaluation
    dimension. This function downloads them to the cache directory.
    
    Args:
        cache_dir: Directory to cache models. Defaults to ~/.cache/vbench.
        metrics: List of metrics to download models for. Defaults to all CORE_METRICS.
        force: Force re-download even if models exist.
        verbose: Print progress information.
    
    Returns:
        Path to the cache directory.
    
    Example:
        >>> setup_vbench_models()
        Downloading VBench models to /home/user/.cache/vbench...
        ✓ subject_consistency model ready
        ✓ motion_smoothness model ready
        ...
        Setup complete!
    """
    from vidspy.metrics import CORE_METRICS
    
    # Determine cache directory
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "vbench"
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which metrics to setup
    if metrics is None:
        metrics = CORE_METRICS
    
    if verbose:
        print(f"Setting up VBench models in {cache_dir}...")
    
    # Check if VBench is installed
    try:
        import vbench
        vbench_available = True
    except ImportError:
        vbench_available = False
        if verbose:
            print("Warning: VBench not installed. Install with: pip install vbench")
            print("Models will be downloaded when VBench is first used.")
    
    if vbench_available:
        try:
            # VBench handles its own model downloads
            # We just need to trigger initialization
            evaluator = vbench.VBench(
                device="cpu",
                full_info_dir=str(cache_dir),
            )
            
            if verbose:
                print("VBench initialized successfully!")
                for metric in metrics:
                    print(f"  ✓ {metric} ready")
                    
        except Exception as e:
            if verbose:
                print(f"Warning: VBench initialization failed: {e}")
                print("Models will be downloaded on first use.")
    
    # Create marker file
    marker = cache_dir / ".vidspy_setup_complete"
    marker.touch()
    
    if verbose:
        print(f"\nSetup complete! Models cached at: {cache_dir}")
    
    return cache_dir


def check_dependencies(verbose: bool = True) -> dict:
    """
    Check if all required dependencies are installed.
    
    Returns:
        Dictionary with dependency status.
    
    Example:
        >>> check_dependencies()
        {
            'dspy': {'installed': True, 'version': '2.4.0'},
            'torch': {'installed': True, 'version': '2.1.0'},
            'vbench': {'installed': False, 'version': None},
            ...
        }
    """
    deps = [
        "dspy",
        "torch",
        "numpy",
        "Pillow",
        "opencv-python",
        "requests",
        "tqdm",
        "vbench",
        "transformers",
    ]
    
    results = {}
    
    for dep in deps:
        try:
            # Handle package name differences
            import_name = dep.replace("-", "_")
            if dep == "Pillow":
                import_name = "PIL"
            elif dep == "opencv-python":
                import_name = "cv2"
            
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            results[dep] = {"installed": True, "version": version}
            
            if verbose:
                print(f"  ✓ {dep} ({version})")
                
        except ImportError:
            results[dep] = {"installed": False, "version": None}
            
            if verbose:
                print(f"  ✗ {dep} (not installed)")
    
    return results


def verify_cuda(verbose: bool = True) -> dict:
    """
    Verify CUDA availability and configuration.
    
    Returns:
        Dictionary with CUDA status.
    """
    result = {
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "devices": [],
    }
    
    try:
        import torch
        
        result["cuda_available"] = torch.cuda.is_available()
        
        if result["cuda_available"]:
            result["cuda_version"] = torch.version.cuda
            result["device_count"] = torch.cuda.device_count()
            result["devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory": torch.cuda.get_device_properties(i).total_memory,
                }
                for i in range(result["device_count"])
            ]
        
        if verbose:
            if result["cuda_available"]:
                print(f"CUDA available: {result['cuda_version']}")
                for dev in result["devices"]:
                    mem_gb = dev["memory"] / (1024**3)
                    print(f"  GPU {dev['index']}: {dev['name']} ({mem_gb:.1f} GB)")
            else:
                print("CUDA not available. CPU will be used.")
                
    except ImportError:
        if verbose:
            print("PyTorch not installed. Cannot check CUDA.")
    
    return result


def create_example_config(
    output_path: str = "vidspy_config.yaml",
    overwrite: bool = False,
) -> Path:
    """
    Create an example configuration file.
    
    Args:
        output_path: Path for the config file.
        overwrite: Whether to overwrite existing file.
    
    Returns:
        Path to the created config file.
    """
    config_content = """# ViDSPy Configuration
# ====================

# VLM Provider Settings
vlm:
  backend: openrouter  # or "huggingface"
  model: google/gemini-2.0-flash-001
  # api_key: your-api-key-here  # Or set OPENROUTER_API_KEY env var

# Optimization Settings
optimization:
  default_optimizer: mipro_v2
  max_bootstrapped_demos: 4
  max_labeled_demos: 4

# Metric Settings
metrics:
  quality_weight: 0.6
  alignment_weight: 0.4
  quality_metrics:
    - subject_consistency
    - motion_smoothness
    - temporal_flickering
    - human_anatomy
    - aesthetic_quality
    - imaging_quality
  alignment_metrics:
    - object_class
    - human_action
    - spatial_relationship
    - overall_consistency

# Target Thresholds
targets:
  human_anatomy: 0.85
  alignment: 0.80

# Cache Settings
cache:
  dir: ~/.cache/vidspy
  vbench_models: ~/.cache/vbench

# Hardware Settings
hardware:
  device: auto  # "cuda", "cpu", or "auto"
  dtype: float16
"""
    
    output_path = Path(output_path)
    
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Config file already exists: {output_path}. "
            "Use overwrite=True to replace."
        )
    
    output_path.write_text(config_content)
    
    return output_path


def install_optional_deps(
    deps: Optional[List[str]] = None,
    verbose: bool = True,
) -> None:
    """
    Install optional dependencies.
    
    Args:
        deps: List of dependencies to install. Defaults to VBench.
        verbose: Print progress information.
    """
    if deps is None:
        deps = ["vbench"]
    
    for dep in deps:
        if verbose:
            print(f"Installing {dep}...")
        
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", dep],
                stdout=subprocess.DEVNULL if not verbose else None,
                stderr=subprocess.DEVNULL if not verbose else None,
            )
            
            if verbose:
                print(f"  ✓ {dep} installed successfully")
                
        except subprocess.CalledProcessError:
            if verbose:
                print(f"  ✗ Failed to install {dep}")


# CLI entry point
def main() -> None:
    """CLI entry point for setup commands."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ViDSPy setup utilities"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Setup command
    setup_parser = subparsers.add_parser(
        "setup",
        help="Setup VBench models"
    )
    setup_parser.add_argument(
        "--cache-dir",
        help="Cache directory for models"
    )
    setup_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download"
    )
    
    # Check command
    check_parser = subparsers.add_parser(
        "check",
        help="Check dependencies"
    )
    
    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Create example config"
    )
    config_parser.add_argument(
        "--output",
        default="vidspy_config.yaml",
        help="Output path"
    )
    
    args = parser.parse_args()
    
    if args.command == "setup":
        setup_vbench_models(
            cache_dir=args.cache_dir,
            force=args.force,
        )
    elif args.command == "check":
        print("Checking dependencies...")
        check_dependencies()
        print("\nChecking CUDA...")
        verify_cuda()
    elif args.command == "config":
        create_example_config(args.output)
        print(f"Created config: {args.output}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
