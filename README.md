# ViDSPy 🎬

**DSPy-style framework for optimizing text-to-video generation via VBench metric feedback.**

[![PyPI version](https://img.shields.io/pypi/v/vidspy)](https://pypi.org/project/vidspy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

ViDSPy brings the power of [DSPy](https://github.com/stanfordnlp/dspy)'s declarative programming paradigm to text-to-video generation. Optimize your video generation prompts and few-shot demonstrations using VBench quality metrics as feedback signals.

## 🎯 Key Features

- **DSPy-style Optimization**: Tune instructions (prompt templates) and demonstrations (few-shot examples) using canonical DSPy optimizers
- **VBench Integration**: Full support for all 10 CORE_METRICS from VBench evaluation
- **Multiple Optimizers**: BootstrapFewShot, LabeledFewShot, MIPROv2, COPRO, and GEPA
- **Flexible VLM Backends**: OpenRouter (cloud) and HuggingFace (local) support
- **Composite Metrics**: Weighted combination of video quality (60%) and text-video alignment (40%)

## 📦 Installation

```bash
pip install vidspy
```

For full functionality with VBench evaluation:

```bash
pip install vidspy[vbench]
```

For development:

```bash
pip install vidspy[all]
```

## ⚙️ Configuration

ViDSPy can be configured in two ways:

### Option 1: Pass Arguments Directly in Code

```python
from vidspy import ViDSPy

vidspy = ViDSPy(
    vlm_backend="openrouter",
    vlm_model="google/gemini-2.0-flash-001",
    api_key="your-api-key",
    device="auto"
)
```

### Option 2: Use Configuration File

Create a `vidspy_config.yaml` file from the template:

```bash
cp vidspy_config.yaml.example vidspy_config.yaml
```

Edit the configuration file:

```yaml
# vidspy_config.yaml
vlm:
  backend: openrouter
  model: google/gemini-2.0-flash-001

optimization:
  default_optimizer: mipro_v2
  max_bootstrapped_demos: 4

metrics:
  quality_weight: 0.6
  alignment_weight: 0.4

cache:
  dir: ~/.cache/vidspy

hardware:
  device: auto
  dtype: float16
```

Then use ViDSPy without any arguments - it automatically loads the config:

```python
from vidspy import ViDSPy, VideoChainOfThought, Example

# ViDSPy automatically finds and loads vidspy_config.yaml
vidspy = ViDSPy()

# All settings from the config file are now applied!
trainset = [Example(prompt="a cat jumping", video_path="cat.mp4")]
optimized = vidspy.optimize(VideoChainOfThought("prompt -> video"), trainset)
```

**Config file search order:**

ViDSPy automatically searches for `vidspy_config.yaml` in:
1. Current working directory: `./vidspy_config.yaml`
2. User config directory: `~/.vidspy/config.yaml`
3. User home directory: `~/vidspy_config.yaml`

**Custom config path:**

You can also specify a custom config file location:

```python
vidspy = ViDSPy(config_path="/path/to/custom_config.yaml")
```

**Important:**
- Arguments passed directly to `ViDSPy()` always override config file values
- API keys should be in environment variables or `.env` file (not in the config file):

```bash
# .env
OPENROUTER_API_KEY=your-api-key-here
```

## 🚀 Quick Start

```python
from vidspy import ViDSPy, VideoChainOfThought, Example

# Initialize ViDSPy with OpenRouter VLM backend
vidspy = ViDSPy(vlm_backend="openrouter")

# Create training examples
trainset = [
    Example(prompt="a cat jumping over a fence", video_path="cat_jump.mp4"),
    Example(prompt="a dog running in a park", video_path="dog_run.mp4"),
    Example(prompt="a bird flying through clouds", video_path="bird_fly.mp4"),
]

# Optimize a video generation module
optimized = vidspy.optimize(
    VideoChainOfThought("prompt -> video"),
    trainset,
    optimizer="mipro_v2"  # Multi-stage instruction + demo optimization
)

# Generate videos with optimized prompts and demonstrations
result = optimized("a dolphin swimming in the ocean")
print(f"Generated video: {result.video_path}")
```

## 📊 VBench Metrics

ViDSPy uses VBench's 10 CORE_METRICS split into two categories:

### Video Quality Metrics (60% weight, video-only)

| Metric | Description |
|--------|-------------|
| `subject_consistency` | Temporal stability of subjects |
| `motion_smoothness` | Natural motion quality |
| `temporal_flickering` | Absence of temporal jitter |
| `human_anatomy` | Correct hands/faces/torso rendering |
| `aesthetic_quality` | Artistic/visual beauty |
| `imaging_quality` | Technical clarity and sharpness |

### Text-Video Alignment Metrics (40% weight, prompt-conditioned)

| Metric | Description |
|--------|-------------|
| `object_class` | Prompt objects appear correctly |
| `human_action` | Prompt actions performed correctly |
| `spatial_relationship` | Correct spatial layout |
| `overall_consistency` | Holistic text-video alignment |

### Using Metrics

```python
from vidspy.metrics import composite_reward, quality_score, alignment_score

# Default composite metric (60% quality + 40% alignment)
score = composite_reward(example, prediction)

# Quality-only score
q_score = quality_score(example, prediction)

# Alignment-only score
a_score = alignment_score(example, prediction)

# Custom metric configuration
from vidspy.metrics import VBenchMetric

custom_metric = VBenchMetric(
    quality_weight=0.5,
    alignment_weight=0.5,
    quality_metrics=["motion_smoothness", "aesthetic_quality"],
    alignment_metrics=["object_class", "overall_consistency"]
)
```

## 🔧 Optimizers

ViDSPy provides 5 DSPy-compatible optimizers:

| Optimizer | Description | Key Parameters |
|-----------|-------------|----------------|
| `VidBootstrapFewShot` | Auto-generate/select few-shots | `max_bootstrapped_demos=4` |
| `VidLabeledFewShot` | Static few-shot assignment | `k=3` |
| `VidMIPROv2` | Multi-stage instruction + demo optimization | `num_candidates=10, auto="light"` |
| `VidCOPRO` | Cooperative multi-LM instruction optimization | `breadth=5, depth=3` |
| `VidGEPA` | Generate + Evaluate + Propose + Accept | `auto="light"` |

### Example: Using Different Optimizers

```python
# Bootstrap few-shot
optimized = vidspy.optimize(
    module, trainset,
    optimizer="bootstrap",
    max_bootstrapped_demos=4
)

# MIPROv2 with more candidates
optimized = vidspy.optimize(
    module, trainset,
    optimizer="mipro_v2",
    num_candidates=15,
    auto="medium"
)

# COPRO with custom search
optimized = vidspy.optimize(
    module, trainset,
    optimizer="copro",
    breadth=10,
    depth=5
)
```

## 🤖 VLM Providers

### OpenRouter (Default)

Cloud-based video VLMs via unified API:

```python
vidspy = ViDSPy(
    vlm_backend="openrouter",
    vlm_model="google/gemini-2.0-flash-001",
    api_key="your-api-key"  # Or set OPENROUTER_API_KEY env var
)
```

Supported models:
- `google/gemini-2.0-flash-001` (default)
- `google/gemini-1.5-pro`
- `anthropic/claude-3-opus`
- `openai/gpt-4o`

### HuggingFace (Local)

Local video VLMs for offline usage:

```python
vidspy = ViDSPy(
    vlm_backend="huggingface",
    vlm_model="llava-hf/llava-v1.6-mistral-7b-hf",
    device="cuda"
)
```

## 📁 Video Modules

ViDSPy provides several module types for different use cases:

```python
from vidspy import VideoPredict, VideoChainOfThought, VideoReAct, VideoEnsemble

# Simple prediction
predictor = VideoPredict("prompt -> video_path")

# Chain-of-thought reasoning
cot = VideoChainOfThought("prompt -> video")

# ReAct-style iterative refinement
react = VideoReAct("prompt -> video", max_iterations=3)

# Ensemble multiple approaches
ensemble = VideoEnsemble([
    VideoPredict(),
    VideoChainOfThought(),
], selection_metric=composite_reward)
```

## 🛠️ Setup VBench Models

```bash
# Via CLI
vidspy setup

# Via Python
from vidspy import setup_vbench_models
setup_vbench_models()  # Downloads to ~/.cache/vbench
```

## 📝 Full Example

```python
import os
from vidspy import (
    ViDSPy,
    VideoChainOfThought,
    Example,
    composite_reward,
)

# Set API key
os.environ["OPENROUTER_API_KEY"] = "your-api-key"

# Initialize
vidspy = ViDSPy(vlm_backend="openrouter")

# Prepare training data
trainset = [
    Example(
        prompt="a person walking through a forest",
        video_path="data/walk_forest.mp4"
    ),
    Example(
        prompt="a car driving on a highway",
        video_path="data/car_highway.mp4"
    ),
    # ... more examples
]

# Split for validation
valset = trainset[-2:]
trainset = trainset[:-2]

# Create and optimize module
module = VideoChainOfThought("prompt -> video")

optimized = vidspy.optimize(
    module,
    trainset,
    valset=valset,
    metric=composite_reward,
    optimizer="mipro_v2",
    num_candidates=10,
)

# Evaluate on test set
testset = [Example(prompt="a boat on a lake", video_path="data/boat.mp4")]
results = vidspy.evaluate(optimized, testset)

print(f"Mean Score: {results['mean_score']:.4f}")
print(f"Quality: {results['details'][0].get('quality_score', 'N/A')}")
print(f"Alignment: {results['details'][0].get('alignment_score', 'N/A')}")

# Generate new videos
result = optimized("a butterfly landing on a flower")
print(f"Generated: {result.video_path}")
print(f"Enhanced prompt: {result.enhanced_prompt}")
```

## 📖 CLI Reference

```bash
# Show help
vidspy --help

# Setup VBench models
vidspy setup
vidspy setup --cache-dir /path/to/cache

# Check dependencies
vidspy setup --check-only

# Optimize a module
vidspy optimize trainset.json --optimizer mipro_v2 --output optimized_model

# Evaluate a module
vidspy evaluate testset.json --module optimized_model --output results.json

# Show information
vidspy info
```

## 🏗️ Project Structure

```
vidspy/
├── pyproject.toml              # Package configuration
├── README.md                   # This file
├── .env.example                # Environment variables template
├── vidspy_config.yaml.example  # Configuration file template
├── vidspy/
│   ├── __init__.py             # Main exports
│   ├── core.py                 # ViDSPy main class, Example
│   ├── signatures.py           # VideoSignature, etc.
│   ├── modules.py              # VideoPredict, VideoChainOfThought
│   ├── optimizers.py           # VidBootstrapFewShot, VidMIPROv2, etc.
│   ├── metrics.py              # VBench wrappers, composite_reward
│   ├── providers.py            # OpenRouterVLM, HuggingFaceVLM
│   ├── setup.py                # Setup utilities
│   └── cli.py                  # Command-line interface
├── examples/
│   └── basic_usage.py
└── tests/
    └── test_basic.py
```

## 🔬 Target Thresholds

For production-quality videos, aim for:

- **Human Anatomy**: ≥ 0.85
- **Text-Video Alignment**: ≥ 0.80

## 📚 References

- [DSPy](https://github.com/stanfordnlp/dspy) - Declarative Self-improving Python
- [VBench](https://github.com/Vchitect/VBench) - Video generation benchmark
- [OpenRouter](https://openrouter.ai/) - Unified AI API

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions welcome! Please read our [contributing guide](CONTRIBUTING.md) first.

## ⭐ Star History

If you find ViDSPy useful, please consider giving it a star!
