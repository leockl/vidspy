"""
Core ViDSPy class and data structures.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import dspy


@dataclass
class Example:
    """
    A training example for video optimization.
    
    Attributes:
        prompt: Text prompt describing the desired video content.
        video_path: Path to the video file (can be local or URL).
        metadata: Optional additional metadata for the example.
    """
    prompt: str
    video_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
    
    def to_dspy_example(self) -> dspy.Example:
        """Convert to DSPy Example format."""
        return dspy.Example(
            prompt=self.prompt,
            video_path=self.video_path,
            **self.metadata
        ).with_inputs("prompt")
    
    @classmethod
    def from_dspy_example(cls, example: dspy.Example) -> "Example":
        """Create from DSPy Example."""
        return cls(
            prompt=example.prompt,
            video_path=example.video_path,
            metadata={k: v for k, v in example.items() if k not in ("prompt", "video_path")}
        )


@dataclass
class VideoExample(Example):
    """
    Extended video example with additional video-specific attributes.
    
    Attributes:
        prompt: Text prompt describing the desired video content.
        video_path: Path to the video file.
        duration: Video duration in seconds.
        fps: Frames per second.
        resolution: Video resolution as (width, height).
        quality_scores: Pre-computed quality scores.
        alignment_scores: Pre-computed alignment scores.
    """
    duration: Optional[float] = None
    fps: Optional[float] = None
    resolution: Optional[tuple[int, int]] = None
    quality_scores: Dict[str, float] = field(default_factory=dict)
    alignment_scores: Dict[str, float] = field(default_factory=dict)


class ViDSPy:
    """
    Main ViDSPy framework class for optimizing text-to-video generation.

    ViDSPy provides DSPy-style optimization of video generation prompts and
    few-shot demonstrations using VBench metrics as feedback signals.

    Args:
        vlm_backend: VLM backend to use ("openrouter" or "huggingface").
                    If None, uses value from config file or defaults to "openrouter".
        vlm_model: Specific VLM model identifier.
                  If None, uses value from config file or backend defaults.
        api_key: API key for OpenRouter (if using OpenRouter backend).
                If None, checks config file then OPENROUTER_API_KEY env var.
        optimizer_lm: Language model for DSPy optimizers (e.g., "openai/gpt-4o-mini").
                     This LLM is used by MIPROv2, COPRO, GEPA to generate instruction variations.
                     If None, uses value from config file or defaults to "openai/gpt-4o-mini".
        optimizer_api_key: API key for optimizer LLM via OpenRouter.
                          If None, checks config then OPENROUTER_OPTIMIZER_API_KEY env var.
        cache_dir: Directory for caching models and results.
                  If None, uses value from config file or defaults to ~/.cache/vidspy.
        device: Device for computation ("cuda", "cpu", or "auto").
               If None, uses value from config file or defaults to "auto".
        config_path: Path to YAML configuration file. If None, searches for
                    vidspy_config.yaml in current directory then home directory.
    
    Example:
        >>> vidspy = ViDSPy(vlm_backend="openrouter")
        >>> trainset = [Example(prompt="cat jumping", video_path="cat.mp4")]
        >>> optimized = vidspy.optimize(
        ...     VideoChainOfThought("prompt -> video"),
        ...     trainset,
        ...     optimizer="mipro_v2"
        ... )
    """

    # Available optimizer names
    AVAILABLE_OPTIMIZERS = ["bootstrap", "labeled", "mipro_v2", "copro", "gepa"]

    def __init__(
        self,
        vlm_backend: Optional[Literal["openrouter", "huggingface"]] = None,
        vlm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        optimizer_lm: Optional[str] = None,
        optimizer_api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: Optional[Literal["cuda", "cpu", "auto"]] = None,
        config_path: Optional[str] = None,
    ) -> None:
        # Load configuration from YAML file if it exists
        from vidspy.setup import load_config
        config = load_config(config_path)

        # Extract config sections
        vlm_config = config.get("vlm", {})
        optimizer_config = config.get("optimizer", {})
        cache_config = config.get("cache", {})
        hardware_config = config.get("hardware", {})

        # Merge config with arguments (arguments take precedence)
        self.vlm_backend = vlm_backend or vlm_config.get("backend", "openrouter")
        self.vlm_model = vlm_model or vlm_config.get("model")
        self.api_key = api_key or vlm_config.get("api_key") or os.environ.get("OPENROUTER_API_KEY")

        # Optimizer LLM configuration
        self.optimizer_lm = optimizer_lm or optimizer_config.get("lm", "openai/gpt-4o-mini")
        self.optimizer_api_key = (
            optimizer_api_key or
            optimizer_config.get("api_key") or
            os.environ.get("OPENROUTER_OPTIMIZER_API_KEY") or
            self.api_key  # Fallback to VLM API key if not specified
        )

        # Cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        elif cache_config.get("dir"):
            self.cache_dir = Path(cache_config["dir"]).expanduser()
        else:
            self.cache_dir = Path.home() / ".cache" / "vidspy"

        # Device
        self.device = device or hardware_config.get("device", "auto")

        # Store full config for later use
        self._config = config

        # Initialize VLM provider
        self._vlm = self._init_vlm()

        # Track if DSPy optimizer LLM has been configured
        self._dspy_configured = False

        # Initialize metric calculator
        self._metric = None
        
    def _init_vlm(self) -> Any:
        """Initialize the VLM provider based on backend selection."""
        from vidspy.providers import OpenRouterVLM, HuggingFaceVLM

        if self.vlm_backend == "openrouter":
            return OpenRouterVLM(
                model=self.vlm_model or "google/gemini-2.5-flash",
                api_key=self.api_key,
            )
        elif self.vlm_backend == "huggingface":
            return HuggingFaceVLM(
                model=self.vlm_model or "llava-hf/llava-v1.6-mistral-7b-hf",
                device=self.device,
                cache_dir=str(self.cache_dir / "models"),
            )
        else:
            raise ValueError(f"Unknown VLM backend: {self.vlm_backend}")

    def _configure_dspy_optimizer(self) -> None:
        """
        Configure DSPy with the optimizer LLM via OpenRouter.

        This is called lazily only when using optimizers that need an LLM
        (MIPROv2, COPRO, GEPA).
        """
        if self._dspy_configured:
            return  # Already configured

        if not self.optimizer_api_key:
            raise ValueError(
                "Optimizer API key required for this optimizer. "
                "Set OPENROUTER_OPTIMIZER_API_KEY env var, "
                "provide optimizer_api_key parameter, or configure in vidspy_config.yaml.\n"
                "Note: BootstrapFewShot and LabeledFewShot do not require this."
            )

        try:
            # Configure DSPy LM using OpenRouter
            lm = dspy.LM(
                model=f"openrouter/{self.optimizer_lm}",
                api_key=self.optimizer_api_key,
                api_base="https://openrouter.ai/api/v1"
            )
            dspy.configure(lm=lm)
            self._dspy_configured = True
        except Exception as e:
            raise RuntimeError(
                f"Failed to configure DSPy optimizer LLM: {e}\n"
                f"Model: {self.optimizer_lm}\n"
                "Ensure the model is available on OpenRouter and API key is valid."
            )

    def _get_optimizer(
        self,
        optimizer: str,
        metric: Callable,
        **kwargs: Any,
    ) -> Any:
        """Get the appropriate optimizer instance."""
        from vidspy.optimizers import (
            VidBootstrapFewShot,
            VidLabeledFewShot,
            VidMIPROv2,
            VidCOPRO,
            VidGEPA,
        )
        
        optimizers = {
            "bootstrap": lambda: VidBootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=kwargs.get("max_bootstrapped_demos", 4),
                max_labeled_demos=kwargs.get("max_labeled_demos", 4),
                **{k: v for k, v in kwargs.items() 
                   if k not in ("max_bootstrapped_demos", "max_labeled_demos")}
            ),
            "labeled": lambda: VidLabeledFewShot(
                metric=metric,
                k=kwargs.get("k", 3),
                **{k: v for k, v in kwargs.items() if k != "k"}
            ),
            "mipro_v2": lambda: VidMIPROv2(
                metric=metric,
                num_candidates=kwargs.get("num_candidates", 10),
                auto=kwargs.get("auto", "light"),
                **{k: v for k, v in kwargs.items() 
                   if k not in ("num_candidates", "auto")}
            ),
            "copro": lambda: VidCOPRO(
                metric=metric,
                breadth=kwargs.get("breadth", 5),
                depth=kwargs.get("depth", 3),
                **{k: v for k, v in kwargs.items() 
                   if k not in ("breadth", "depth")}
            ),
            "gepa": lambda: VidGEPA(
                metric=metric,
                auto=kwargs.get("auto", "light"),
                **{k: v for k, v in kwargs.items() if k != "auto"}
            ),
        }
        
        if optimizer not in optimizers:
            raise ValueError(
                f"Unknown optimizer: {optimizer}. "
                f"Available: {list(optimizers.keys())}"
            )
        
        return optimizers[optimizer]()
    
    def optimize(
        self,
        module: Any,
        trainset: List[Union[Example, dspy.Example]],
        metric: Optional[Callable] = None,
        optimizer: str = "mipro_v2",
        valset: Optional[List[Union[Example, dspy.Example]]] = None,
        **optimizer_kwargs: Any,
    ) -> Any:
        """
        Optimize a video generation module using the specified optimizer.
        
        Args:
            module: The video module to optimize (e.g., VideoChainOfThought).
            trainset: List of training examples.
            metric: Custom metric function. Defaults to composite_reward.
            optimizer: Optimizer to use. One of:
                - "bootstrap": VidBootstrapFewShot
                - "labeled": VidLabeledFewShot  
                - "mipro_v2": VidMIPROv2 (default)
                - "copro": VidCOPRO
                - "gepa": VidGEPA
            valset: Optional validation set for early stopping.
            **optimizer_kwargs: Additional arguments for the optimizer.
        
        Returns:
            Optimized module with tuned instructions and demonstrations.
        
        Example:
            >>> optimized = vidspy.optimize(
            ...     VideoChainOfThought("prompt -> video"),
            ...     trainset,
            ...     optimizer="mipro_v2",
            ...     num_candidates=15
            ... )
        """
        from vidspy.metrics import composite_reward

        # Use default metric if not provided
        if metric is None:
            metric = composite_reward

        # Configure DSPy optimizer LLM if using an optimizer that needs it
        # MIPROv2, COPRO, and GEPA use LLMs for instruction generation
        # BootstrapFewShot and LabeledFewShot do not
        llm_requiring_optimizers = ["mipro_v2", "copro", "gepa"]
        if optimizer.lower() in llm_requiring_optimizers:
            self._configure_dspy_optimizer()

        # Convert Examples to DSPy format
        dspy_trainset = [
            ex.to_dspy_example() if isinstance(ex, Example) else ex
            for ex in trainset
        ]

        dspy_valset = None
        if valset is not None:
            dspy_valset = [
                ex.to_dspy_example() if isinstance(ex, Example) else ex
                for ex in valset
            ]

        # Get optimizer instance
        opt = self._get_optimizer(optimizer, metric, **optimizer_kwargs)
        
        # Run optimization
        if dspy_valset is not None:
            optimized = opt.compile(
                module,
                trainset=dspy_trainset,
                valset=dspy_valset,
            )
        else:
            optimized = opt.compile(
                module,
                trainset=dspy_trainset,
            )
        
        return optimized
    
    def evaluate(
        self,
        module: Any,
        testset: List[Union[Example, dspy.Example]],
        metric: Optional[Callable] = None,
        num_threads: int = 1,
        display_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a video module on a test set.
        
        Args:
            module: The video module to evaluate.
            testset: List of test examples.
            metric: Custom metric function. Defaults to composite_reward.
            num_threads: Number of threads for parallel evaluation.
            display_progress: Whether to display progress bar.
        
        Returns:
            Dictionary with evaluation results including:
                - mean_score: Average metric score
                - std_score: Standard deviation
                - scores: List of individual scores
                - details: Per-example breakdown
        """
        from vidspy.metrics import composite_reward
        import numpy as np
        from tqdm import tqdm
        
        if metric is None:
            metric = composite_reward
        
        # Convert Examples to DSPy format
        dspy_testset = [
            ex.to_dspy_example() if isinstance(ex, Example) else ex
            for ex in testset
        ]
        
        scores = []
        details = []
        
        iterator = tqdm(dspy_testset, disable=not display_progress, desc="Evaluating")
        
        for example in iterator:
            try:
                # Run module
                pred = module(prompt=example.prompt)
                
                # Calculate score
                score = metric(example, pred)
                scores.append(score)
                
                details.append({
                    "prompt": example.prompt,
                    "score": score,
                    "success": True,
                })
            except Exception as e:
                scores.append(0.0)
                details.append({
                    "prompt": example.prompt,
                    "score": 0.0,
                    "success": False,
                    "error": str(e),
                })
        
        return {
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "scores": scores,
            "details": details,
            "num_examples": len(scores),
            "num_successes": sum(1 for d in details if d["success"]),
        }
    
    def configure(
        self,
        vlm_backend: Optional[str] = None,
        vlm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        optimizer_lm: Optional[str] = None,
        optimizer_api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
    ) -> "ViDSPy":
        """
        Reconfigure ViDSPy settings.

        Args:
            vlm_backend: VLM backend to use.
            vlm_model: Specific VLM model identifier.
            api_key: API key for OpenRouter.
            optimizer_lm: Language model for DSPy optimizers.
            optimizer_api_key: API key for optimizer LLM.
            cache_dir: Directory for caching.
            device: Device for computation.

        Returns:
            Self for method chaining.
        """
        if vlm_backend is not None:
            self.vlm_backend = vlm_backend
        if vlm_model is not None:
            self.vlm_model = vlm_model
        if api_key is not None:
            self.api_key = api_key
        if optimizer_lm is not None:
            self.optimizer_lm = optimizer_lm
            # Reset DSPy configuration if optimizer LLM changed
            self._dspy_configured = False
        if optimizer_api_key is not None:
            self.optimizer_api_key = optimizer_api_key
            # Reset DSPy configuration if optimizer API key changed
            self._dspy_configured = False
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
        if device is not None:
            self.device = device

        # Reinitialize VLM
        self._vlm = self._init_vlm()

        return self
