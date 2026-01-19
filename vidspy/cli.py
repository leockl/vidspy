"""
Command-line interface for ViDSPy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def cmd_setup(args: argparse.Namespace) -> int:
    """Run the setup command."""
    from vidspy.setup import setup_vbench_models, check_dependencies, verify_cuda
    
    print("=" * 60)
    print("ViDSPy Setup")
    print("=" * 60)
    
    if args.check_only:
        print("\n📦 Checking dependencies...")
        check_dependencies()
        print("\n🖥️  Checking CUDA...")
        verify_cuda()
        return 0
    
    print("\n📦 Checking dependencies...")
    check_dependencies()
    
    print("\n🖥️  Checking CUDA...")
    verify_cuda()
    
    print("\n📥 Setting up VBench models...")
    setup_vbench_models(
        cache_dir=args.cache_dir,
        force=args.force,
    )
    
    print("\n✅ Setup complete!")
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    """Run optimization command."""
    from vidspy import ViDSPy, Example, VideoChainOfThought
    from vidspy.metrics import composite_reward
    
    print(f"Loading training data from {args.trainset}...")
    
    # Load training data
    trainset_path = Path(args.trainset)
    if not trainset_path.exists():
        print(f"Error: Training set not found: {args.trainset}")
        return 1
    
    with open(trainset_path) as f:
        train_data = json.load(f)
    
    trainset = [
        Example(prompt=ex["prompt"], video_path=ex["video_path"])
        for ex in train_data
    ]
    
    print(f"Loaded {len(trainset)} training examples")
    
    # Initialize ViDSPy
    vidspy = ViDSPy(
        vlm_backend=args.backend,
        vlm_model=args.model,
        api_key=args.api_key,
    )
    
    # Create module
    module = VideoChainOfThought("prompt -> video")
    
    # Run optimization
    print(f"\nOptimizing with {args.optimizer}...")
    optimized = vidspy.optimize(
        module,
        trainset,
        optimizer=args.optimizer,
    )
    
    # Save optimized module
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # DSPy modules can be saved
    optimized.save(str(output_path))
    
    print(f"\n✅ Optimization complete!")
    print(f"Saved optimized module to: {output_path}")
    
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run evaluation command."""
    import json
    from vidspy import ViDSPy, Example
    from vidspy.metrics import composite_reward
    
    print(f"Loading test data from {args.testset}...")
    
    # Load test data
    testset_path = Path(args.testset)
    if not testset_path.exists():
        print(f"Error: Test set not found: {args.testset}")
        return 1
    
    with open(testset_path) as f:
        test_data = json.load(f)
    
    testset = [
        Example(prompt=ex["prompt"], video_path=ex["video_path"])
        for ex in test_data
    ]
    
    print(f"Loaded {len(testset)} test examples")
    
    # Initialize ViDSPy
    vidspy = ViDSPy(
        vlm_backend=args.backend,
        vlm_model=args.model,
    )
    
    # Load module if provided
    if args.module:
        import dspy
        module = dspy.Module.load(args.module)
    else:
        from vidspy import VideoChainOfThought
        module = VideoChainOfThought("prompt -> video")
    
    # Run evaluation
    print("\nEvaluating...")
    results = vidspy.evaluate(
        module,
        testset,
        display_progress=True,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Mean Score:  {results['mean_score']:.4f}")
    print(f"Std Score:   {results['std_score']:.4f}")
    print(f"Min Score:   {results['min_score']:.4f}")
    print(f"Max Score:   {results['max_score']:.4f}")
    print(f"Success Rate: {results['num_successes']}/{results['num_examples']}")
    
    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show ViDSPy information."""
    from vidspy import __version__
    from vidspy.metrics import CORE_METRICS, QUALITY_METRICS, ALIGNMENT_METRICS
    from vidspy.providers import OpenRouterVLM
    
    print("=" * 60)
    print(f"ViDSPy v{__version__}")
    print("=" * 60)
    print("\nDSPy-style framework for text-to-video optimization")
    print("using VBench metric feedback.\n")
    
    print("📊 CORE_METRICS (10 high-impact metrics):")
    print("\n  Video Quality (60% weight):")
    for m in QUALITY_METRICS:
        print(f"    • {m}")
    
    print("\n  Text-Video Alignment (40% weight):")
    for m in ALIGNMENT_METRICS:
        print(f"    • {m}")
    
    print("\n🔧 Optimizers:")
    optimizers = [
        ("bootstrap", "Auto-generate/select few-shots"),
        ("labeled", "Static few-shot assignment"),
        ("mipro_v2", "Multi-stage instruction + demo optimization"),
        ("copro", "Cooperative multi-LM instruction optimization"),
        ("gepa", "Generate + Evaluate + Propose + Accept"),
    ]
    for name, desc in optimizers:
        print(f"    • {name}: {desc}")
    
    print("\n🤖 VLM Providers:")
    print("    • openrouter: Cloud-based video VLMs")
    print("    • huggingface: Local video VLMs")
    
    print("\n📚 Video-capable models (OpenRouter):")
    for model in OpenRouterVLM.VIDEO_CAPABLE_MODELS[:5]:
        print(f"    • {model}")
    print("    ...")
    
    print("\n🔗 Links:")
    print("    • GitHub: https://github.com/yourusername/vidspy")
    print("    • Docs: https://github.com/yourusername/vidspy#readme")
    
    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="vidspy",
        description="ViDSPy: DSPy-style text-to-video optimization"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands"
    )
    
    # Setup command
    setup_parser = subparsers.add_parser(
        "setup",
        help="Setup VBench models and check dependencies"
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
    setup_parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check dependencies, don't download"
    )
    setup_parser.set_defaults(func=cmd_setup)
    
    # Optimize command
    opt_parser = subparsers.add_parser(
        "optimize",
        help="Optimize a video generation module"
    )
    opt_parser.add_argument(
        "trainset",
        help="Path to training set JSON file"
    )
    opt_parser.add_argument(
        "--output", "-o",
        default="optimized_module",
        help="Output path for optimized module"
    )
    opt_parser.add_argument(
        "--optimizer",
        default="mipro_v2",
        choices=["bootstrap", "labeled", "mipro_v2", "copro", "gepa"],
        help="Optimizer to use"
    )
    opt_parser.add_argument(
        "--backend",
        default="openrouter",
        choices=["openrouter", "huggingface"],
        help="VLM backend"
    )
    opt_parser.add_argument(
        "--model",
        help="VLM model identifier"
    )
    opt_parser.add_argument(
        "--api-key",
        help="API key for OpenRouter"
    )
    opt_parser.set_defaults(func=cmd_optimize)
    
    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a video generation module"
    )
    eval_parser.add_argument(
        "testset",
        help="Path to test set JSON file"
    )
    eval_parser.add_argument(
        "--module", "-m",
        help="Path to saved module (optional)"
    )
    eval_parser.add_argument(
        "--output", "-o",
        help="Output path for results JSON"
    )
    eval_parser.add_argument(
        "--backend",
        default="openrouter",
        choices=["openrouter", "huggingface"],
        help="VLM backend"
    )
    eval_parser.add_argument(
        "--model",
        help="VLM model identifier"
    )
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show ViDSPy information"
    )
    info_parser.set_defaults(func=cmd_info)
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
