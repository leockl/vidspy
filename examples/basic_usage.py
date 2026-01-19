#!/usr/bin/env python3
"""
ViDSPy Basic Usage Example

This example demonstrates how to use ViDSPy to optimize
text-to-video generation using VBench metrics.
"""

import os
from pathlib import Path

# Set your OpenRouter API key (or use environment variable)
# os.environ["OPENROUTER_API_KEY"] = "your-api-key-here"

from vidspy import (
    ViDSPy,
    VideoChainOfThought,
    VideoPredict,
    Example,
    composite_reward,
)


def create_sample_trainset():
    """Create a sample training set for demonstration."""
    # In practice, you would have actual video files
    # This creates mock examples for demonstration
    return [
        Example(
            prompt="a cat jumping over a fence in slow motion",
            video_path="/path/to/cat_jump.mp4",
            metadata={"category": "animals", "duration": 5.0}
        ),
        Example(
            prompt="a person walking through a snowy forest",
            video_path="/path/to/walk_forest.mp4",
            metadata={"category": "nature", "duration": 8.0}
        ),
        Example(
            prompt="a car driving on a coastal highway at sunset",
            video_path="/path/to/car_drive.mp4",
            metadata={"category": "vehicles", "duration": 6.0}
        ),
        Example(
            prompt="a bird flying over a mountain lake",
            video_path="/path/to/bird_fly.mp4",
            metadata={"category": "nature", "duration": 4.0}
        ),
        Example(
            prompt="a chef preparing sushi in a restaurant",
            video_path="/path/to/chef_cook.mp4",
            metadata={"category": "food", "duration": 10.0}
        ),
    ]


def example_basic_optimization():
    """Basic optimization example using MIPROv2."""
    print("=" * 60)
    print("Example 1: Basic Optimization with MIPROv2")
    print("=" * 60)
    
    # Initialize ViDSPy
    # Uses OpenRouter by default for cloud-based VLM
    vidspy = ViDSPy(vlm_backend="openrouter")
    
    # Create training set
    trainset = create_sample_trainset()
    
    # Create a video generation module
    module = VideoChainOfThought("prompt -> video")
    
    # Optimize the module
    print("\nOptimizing module...")
    optimized = vidspy.optimize(
        module,
        trainset,
        optimizer="mipro_v2",
        num_candidates=10,
    )
    
    # Generate a video with the optimized module
    print("\nGenerating video with optimized module...")
    result = optimized("a dolphin swimming in crystal clear water")
    
    print(f"\nResults:")
    print(f"  Video path: {result.video_path}")
    print(f"  Enhanced prompt: {result.enhanced_prompt}")
    if result.reasoning:
        print(f"  Reasoning: {result.reasoning[:200]}...")


def example_different_optimizers():
    """Compare different optimizers."""
    print("\n" + "=" * 60)
    print("Example 2: Comparing Different Optimizers")
    print("=" * 60)
    
    vidspy = ViDSPy(vlm_backend="openrouter")
    trainset = create_sample_trainset()
    
    optimizers = [
        ("bootstrap", {"max_bootstrapped_demos": 4}),
        ("labeled", {"k": 3}),
        ("mipro_v2", {"num_candidates": 10, "auto": "light"}),
    ]
    
    for opt_name, opt_kwargs in optimizers:
        print(f"\n--- {opt_name} ---")
        
        module = VideoChainOfThought("prompt -> video")
        
        optimized = vidspy.optimize(
            module,
            trainset,
            optimizer=opt_name,
            **opt_kwargs
        )
        
        result = optimized("a robot dancing in a neon-lit city")
        print(f"  Generated: {result.video_path}")


def example_custom_metric():
    """Use a custom metric configuration."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Metric Configuration")
    print("=" * 60)
    
    from vidspy.metrics import VBenchMetric
    
    # Create a custom metric focused on motion quality
    motion_metric = VBenchMetric(
        quality_weight=0.7,
        alignment_weight=0.3,
        quality_metrics=["motion_smoothness", "temporal_flickering"],
        alignment_metrics=["human_action", "overall_consistency"],
    )
    
    vidspy = ViDSPy(vlm_backend="openrouter")
    trainset = create_sample_trainset()
    
    module = VideoChainOfThought("prompt -> video")
    
    print("\nOptimizing with custom motion-focused metric...")
    optimized = vidspy.optimize(
        module,
        trainset,
        metric=motion_metric,
        optimizer="mipro_v2",
    )
    
    result = optimized("a gymnast performing a backflip")
    print(f"\nGenerated: {result.video_path}")


def example_evaluation():
    """Evaluate a module on a test set."""
    print("\n" + "=" * 60)
    print("Example 4: Evaluating a Module")
    print("=" * 60)
    
    vidspy = ViDSPy(vlm_backend="openrouter")
    
    # Create train and test sets
    all_examples = create_sample_trainset()
    trainset = all_examples[:3]
    testset = all_examples[3:]
    
    # Optimize
    module = VideoChainOfThought("prompt -> video")
    optimized = vidspy.optimize(module, trainset, optimizer="bootstrap")
    
    # Evaluate
    print("\nEvaluating on test set...")
    results = vidspy.evaluate(optimized, testset)
    
    print(f"\nEvaluation Results:")
    print(f"  Mean Score: {results['mean_score']:.4f}")
    print(f"  Std Score:  {results['std_score']:.4f}")
    print(f"  Min Score:  {results['min_score']:.4f}")
    print(f"  Max Score:  {results['max_score']:.4f}")
    print(f"  Success Rate: {results['num_successes']}/{results['num_examples']}")


def example_local_vlm():
    """Use a local HuggingFace VLM."""
    print("\n" + "=" * 60)
    print("Example 5: Using Local HuggingFace VLM")
    print("=" * 60)
    
    # Use local VLM (requires GPU recommended)
    vidspy = ViDSPy(
        vlm_backend="huggingface",
        vlm_model="llava-hf/llava-v1.6-mistral-7b-hf",
        device="auto",  # Uses CUDA if available
    )
    
    trainset = create_sample_trainset()[:2]  # Use fewer examples
    
    module = VideoPredict("prompt -> video")
    
    print("\nOptimizing with local VLM...")
    optimized = vidspy.optimize(
        module,
        trainset,
        optimizer="labeled",
        k=2,
    )
    
    result = optimized("a waterfall in a tropical jungle")
    print(f"\nGenerated: {result.video_path}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ViDSPy Examples")
    print("=" * 60)
    
    # Check if API key is set
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("\n⚠️  OPENROUTER_API_KEY not set!")
        print("Set your API key to run cloud VLM examples:")
        print('  export OPENROUTER_API_KEY="your-api-key"')
        print("\nRunning with mock mode...\n")
    
    # Run examples
    try:
        example_basic_optimization()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    try:
        example_different_optimizers()
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    try:
        example_custom_metric()
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    try:
        example_evaluation()
    except Exception as e:
        print(f"Example 4 failed: {e}")
    
    # Skip local VLM example by default (requires large download)
    # example_local_vlm()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
