# ViDSPy 🎬

**DSPy-style framework for optimizing text-to-video prompts via VBench metric feedback.**

[![PyPI version](https://img.shields.io/pypi/v/vidspy)](https://pypi.org/project/vidspy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

ViDSPy brings the power of [DSPy](https://github.com/stanfordnlp/dspy)'s declarative programming paradigm to text-to-video generation. **Optimize your prompts** for any text-to-video API (Runway, Pika, Replicate, etc.) using VBench quality metrics and VLM-based feedback.

**Note:** ViDSPy optimizes *how you prompt* video generation models. It does not generate videos itself - you bring your own text-to-video API.

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
    vlm_model="google/gemini-2.5-flash",
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
  model: google/gemini-2.5-flash

optimizer:
  lm: openai/gpt-4o-mini  # LLM for generating instruction variations

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
OPENROUTER_OPTIMIZER_API_KEY=your-api-key-here  # For optimizer LLM
```

### 🤖 Optimizer LLM Configuration

ViDSPy uses **two different LLMs**:
1. **VLM (Vision Language Model)**: Analyzes videos and enhances prompts during inference
2. **Optimizer LLM**: Used by **MIPROv2, COPRO, and GEPA** optimizers to generate instruction variations during optimization

**Note:** BootstrapFewShot and LabeledFewShot optimizers do **NOT** require an optimizer LLM - they work without this configuration.

**Configure the Optimizer LLM:**

```python
from vidspy import ViDSPy

vidspy = ViDSPy(
    vlm_backend="openrouter",
    vlm_model="google/gemini-2.5-flash",          # For video analysis
    optimizer_lm="openai/gpt-4o-mini",            # For optimization
    optimizer_api_key="your-openrouter-api-key"   # Or set OPENROUTER_OPTIMIZER_API_KEY
)
```

Or in `vidspy_config.yaml`:

```yaml
vlm:
  backend: openrouter
  model: google/gemini-2.5-flash

optimizer:
  lm: openai/gpt-4o-mini  # Any OpenRouter model
```

**Choosing an Optimizer LLM:**
- **For most users**: `openai/gpt-4o-mini` (fast and cost-effective)
- **For better quality**: `openai/gpt-4o` or `anthropic/claude-3-5-sonnet`
- **Any model from [OpenRouter](https://openrouter.ai/models)** is supported

**Note:** If `optimizer_api_key` is not specified, it will use the VLM API key as fallback.

## 🎥 Connecting Video Generation Models

**Important:** ViDSPy is a **prompt optimization framework** that sits on top of existing text-to-video models. It does NOT generate videos itself. Instead, it optimizes how you prompt external video generation services.

### How ViDSPy Works

```
User Prompt → ViDSPy (optimize prompt) → Text-to-Video API → Generated Video
                ↓
            VBench + VLM (evaluate quality)
                ↓
        Learn better prompting strategies
```

### Setting Up Your Video Generator

You need to provide a `video_generator` function that connects to your preferred text-to-video service:

#### Example 1: Runway Gen-3

```python
from vidspy import VideoChainOfThought

def runway_generator(prompt: str, **kwargs) -> str:
    """Generate video using Runway Gen-3 API."""
    import requests

    response = requests.post(
        "https://api.runwayml.com/v1/generate",
        headers={"Authorization": f"Bearer {RUNWAY_API_KEY}"},
        json={
            "prompt": prompt,
            "model": "gen3",
            "duration": kwargs.get("duration", 5)
        }
    )

    video_url = response.json()["output"]["url"]
    # Download and save video locally
    video_path = f"outputs/{response.json()['id']}.mp4"
    # ... download logic ...
    return video_path

# Create module with your generator
module = VideoChainOfThought(
    "prompt -> video",
    video_generator=runway_generator
)
```

#### Example 2: Replicate (Stable Video Diffusion, CogVideo, etc.)

```python
import replicate

def replicate_generator(prompt: str, **kwargs) -> str:
    """Generate video using Replicate API."""
    output = replicate.run(
        "stability-ai/stable-video-diffusion",
        input={"prompt": prompt}
    )

    # Save video from output URL
    video_path = f"outputs/{uuid.uuid4()}.mp4"
    # ... download logic ...
    return video_path

module = VideoChainOfThought(
    "prompt -> video",
    video_generator=replicate_generator
)
```

#### Example 3: Pika Labs

```python
from pika import PikaClient

def pika_generator(prompt: str, **kwargs) -> str:
    """Generate video using Pika Labs API."""
    client = PikaClient(api_key=PIKA_API_KEY)

    video = client.generate_video(
        prompt=prompt,
        aspect_ratio="16:9",
        duration=3
    )

    return video.download_path

module = VideoChainOfThought(
    "prompt -> video",
    video_generator=pika_generator
)
```

### Supported Text-to-Video Services

ViDSPy works with any text-to-video API. Popular options include:

| Service | API Available | Notes |
|---------|---------------|-------|
| Runway Gen-3 | ✅ Yes | High quality, good motion |
| Pika Labs | ✅ Yes | Creative effects, good for social media |
| Stability AI Video | ✅ Yes | Open weights available |
| Replicate | ✅ Yes | Multiple models (CogVideo, SVD, etc.) |
| LumaAI Dream Machine | ✅ Yes | Cinematic quality |
| HaiperAI | ✅ Yes | Fast generation |
| Morph Studio | ✅ Yes | Style control |

### Custom Video Generator Template

```python
def my_video_generator(prompt: str, **kwargs) -> str:
    """
    Your custom video generation function.

    Args:
        prompt: Enhanced prompt from ViDSPy
        **kwargs: Additional parameters (duration, aspect_ratio, etc.)

    Returns:
        Local path to the generated video file
    """
    # 1. Call your text-to-video API
    # 2. Download the generated video
    # 3. Save it locally
    # 4. Return the file path

    video_path = "path/to/generated/video.mp4"
    return video_path
```

## 🚀 Quick Start

```python
from vidspy import ViDSPy, VideoChainOfThought, Example

# Step 1: Define your video generator (connect to your text-to-video API)
def my_video_generator(prompt: str, **kwargs) -> str:
    """Your text-to-video API integration."""
    # Example: Runway, Pika, Replicate, etc.
    import my_video_api
    video = my_video_api.generate(prompt=prompt)
    return video.save_path()

# Step 2: Initialize ViDSPy with OpenRouter VLM backend
# Note: Optimizer LLM defaults to "openai/gpt-4o-mini" via OpenRouter
# You can customize: ViDSPy(vlm_backend="openrouter", optimizer_lm="openai/gpt-4o")
vidspy = ViDSPy(vlm_backend="openrouter")

# Step 3: Create training examples (use videos you've already generated)
trainset = [
    Example(prompt="a cat jumping over a fence", video_path="cat_jump.mp4"),
    Example(prompt="a dog running in a park", video_path="dog_run.mp4"),
    Example(prompt="a bird flying through clouds", video_path="bird_fly.mp4"),
]

# Step 4: Create module with your video generator
module = VideoChainOfThought(
    "prompt -> video",
    video_generator=my_video_generator  # Connect your generator!
)

# Step 5: Optimize prompting strategy
optimized = vidspy.optimize(
    module,
    trainset,
    optimizer="mipro_v2"  # Multi-stage instruction + demo optimization
)

# Step 6: Generate videos with optimized prompts
result = optimized("a dolphin swimming in the ocean")
print(f"Generated video: {result.video_path}")
print(f"Optimized prompt used: {result.enhanced_prompt}")
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

**Vision Language Models (VLMs)** in ViDSPy are used for:
- 📝 **Prompt enhancement** - Improving user prompts before generation
- 🔍 **Video analysis** - Understanding generated video content
- 🎯 **Quality assessment** - Analyzing text-video alignment
- 🧠 **Chain-of-thought reasoning** - Planning video generation strategies

**Note:** VLMs do NOT generate videos. They help optimize the prompts you send to your text-to-video API.

### OpenRouter (Default)

Cloud-based multimodal VLMs via unified API:

```python
vidspy = ViDSPy(
    vlm_backend="openrouter",
    vlm_model="google/gemini-2.5-flash",
    api_key="your-api-key"  # Or set OPENROUTER_API_KEY env var
)
```

**Example models:**
- `google/gemini-2.5-flash` 
- `google/gemini-1.5-pro` 
- `anthropic/claude-opus-4.5` 
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

ViDSPy provides several module types for different prompting strategies.

### Module Signatures

When creating video modules, you'll typically use simple signature strings like `"prompt -> video"` or `"prompt -> video_path"`. The library internally handles all the complex reasoning steps (scene analysis, motion planning, etc.) based on the module type you choose. You don't need to worry about the internal signature details—just use these standard formats and ViDSPy takes care of the rest.

```python
from vidspy import VideoPredict, VideoChainOfThought, VideoReAct, VideoEnsemble

# Define your video generator once
def my_video_gen(prompt, **kwargs):
    # Your text-to-video API call
    return video_path

# Simple prediction with prompt enhancement
# Just use "prompt -> video_path" - ViDSPy handles the internal complexity
predictor = VideoPredict(
    "prompt -> video_path",
    video_generator=my_video_gen
)

# Chain-of-thought reasoning (analyzes scene, motion, style before generating)
# The simple "prompt -> video" signature is all you need
cot = VideoChainOfThought(
    "prompt -> video",
    video_generator=my_video_gen
)

# ReAct-style iterative refinement (generates, evaluates, refines)
# Same simple signature - internal reasoning is handled automatically
react = VideoReAct(
    "prompt -> video",
    video_generator=my_video_gen,
    max_iterations=3
)

# Ensemble multiple approaches (tries different strategies, picks best)
ensemble = VideoEnsemble([
    VideoPredict(video_generator=my_video_gen),
    VideoChainOfThought(video_generator=my_video_gen),
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
import replicate
from vidspy import (
    ViDSPy,
    VideoChainOfThought,
    Example,
    composite_reward,
)

# Set API key
os.environ["OPENROUTER_API_KEY"] = "your-api-key"

# Define video generator (using Replicate as example)
def replicate_video_generator(prompt: str, **kwargs) -> str:
    """Generate video using Replicate API."""
    output = replicate.run(
        "stability-ai/stable-video-diffusion",
        input={"prompt": prompt}
    )

    # Download and save video
    video_path = f"outputs/{hash(prompt)}.mp4"
    # ... download logic ...
    return video_path

# Initialize ViDSPy
vidspy = ViDSPy(vlm_backend="openrouter")

# Prepare training data (videos you've already generated)
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

# Create module with your video generator
module = VideoChainOfThought(
    "prompt -> video",
    video_generator=replicate_video_generator
)

# Optimize prompting strategy
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

# Generate new videos with optimized prompts
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
