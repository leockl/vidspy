"""
Video generation modules that wrap DSPy patterns for text-to-video tasks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import dspy

from vidspy.signatures import (
    VideoSignature,
    VideoChainOfThoughtSignature,
    VideoReActSignature,
    VideoPromptEnhancementSignature,
)


@dataclass
class VideoOutput:
    """
    Output from a video generation module.

    Attributes:
        video_path: Path to the generated video.
        enhanced_prompt: The enhanced prompt used for generation.
        reasoning: Any reasoning or intermediate steps.
        metadata: Additional metadata about the generation.
    """
    video_path: str
    enhanced_prompt: Optional[str] = None
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        """Allow attribute access for metadata fields."""
        if name in self.metadata:
            return self.metadata[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


class VideoModule(dspy.Module):
    """
    Abstract base class for video generation modules.
    
    All video modules inherit from this class and implement the
    forward method for video generation.
    """
    
    def __init__(
        self,
        signature: Optional[Union[str, type]] = None,
        video_generator: Optional[Callable] = None,
    ) -> None:
        """
        Initialize the video module.
        
        Args:
            signature: DSPy signature string or class.
            video_generator: Optional callable for actual video generation.
                           If not provided, uses a mock generator.
        """
        super().__init__()
        self.signature = signature
        self.video_generator = video_generator or self._mock_video_generator

    def _mock_video_generator(self, prompt: str, **kwargs: Any) -> str:
        """Mock video generator for testing."""
        import hashlib
        import os
        
        # Create a deterministic path based on prompt
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        return f"/tmp/vidspy_mock_{prompt_hash}.mp4"
    
    def set_video_generator(self, generator: Callable) -> None:
        """Set the video generator callable."""
        self.video_generator = generator

    def forward(self, prompt: str, **kwargs: Any) -> VideoOutput:
        """
        Generate a video from the given prompt.

        Args:
            prompt: Text description of the desired video.
            **kwargs: Additional generation parameters.

        Returns:
            VideoOutput with the generated video path and metadata.

        Note:
            Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement the forward() method")
    
    def __call__(self, prompt: str, **kwargs: Any) -> VideoOutput:
        """Make the module callable."""
        return self.forward(prompt, **kwargs)


class VideoPredict(VideoModule):
    """
    Simple video prediction module.
    
    Uses a VLM to enhance the prompt, then generates a video.
    This is the simplest video generation pattern.
    
    Example:
        >>> predictor = VideoPredict("prompt -> video_path")
        >>> result = predictor("a cat playing piano")
        >>> print(result.video_path)
    """
    
    def __init__(
        self,
        signature: Union[str, type] = VideoSignature,
        video_generator: Optional[Callable] = None,
        enhance_prompt: bool = True,
    ) -> None:
        """
        Initialize VideoPredict.
        
        Args:
            signature: DSPy signature for the prediction.
            video_generator: Callable for video generation.
            enhance_prompt: Whether to enhance prompts before generation.
        """
        super().__init__(signature, video_generator)
        self.enhance_prompt = enhance_prompt
        
        # Create internal DSPy predictor for prompt enhancement
        if enhance_prompt:
            self._enhancer = dspy.Predict(VideoPromptEnhancementSignature)
        else:
            self._enhancer = None
    
    def forward(self, prompt: str, **kwargs: Any) -> VideoOutput:
        """Generate a video from the prompt."""
        enhanced = prompt
        reasoning = None
        
        # Enhance prompt if enabled
        if self._enhancer is not None:
            try:
                enhancement = self._enhancer(original_prompt=prompt)
                enhanced = enhancement.enhanced_prompt
                reasoning = f"Enhanced from: {prompt}"
            except Exception:
                # Fall back to original prompt on error
                enhanced = prompt
        
        # Generate video
        video_path = self.video_generator(enhanced, **kwargs)
        
        return VideoOutput(
            video_path=video_path,
            enhanced_prompt=enhanced,
            reasoning=reasoning,
            metadata={"original_prompt": prompt},
        )


class VideoChainOfThought(VideoModule):
    """
    Chain-of-thought video generation module.
    
    Uses structured reasoning to plan and generate videos with
    better quality and text alignment. Includes:
    - Scene analysis
    - Motion planning
    - Style selection
    - Final prompt synthesis
    
    Example:
        >>> cot = VideoChainOfThought("prompt -> video")
        >>> result = cot("a dancer performing ballet")
        >>> print(result.reasoning)
    """
    
    def __init__(
        self,
        signature: Union[str, type] = "prompt -> video",
        video_generator: Optional[Callable] = None,
    ) -> None:
        """
        Initialize VideoChainOfThought.
        
        Args:
            signature: Signature string or class (simplified for CoT).
            video_generator: Callable for video generation.
        """
        super().__init__(signature, video_generator)
        
        # Create chain-of-thought predictor
        self._cot = dspy.ChainOfThought(VideoChainOfThoughtSignature)
    
    def forward(self, prompt: str, **kwargs: Any) -> VideoOutput:
        """Generate a video with chain-of-thought reasoning."""
        try:
            # Run chain-of-thought
            result = self._cot(prompt=prompt)
            
            enhanced = getattr(result, "enhanced_prompt", prompt)
            
            # Build reasoning from intermediate steps
            reasoning_parts = []
            if hasattr(result, "scene_analysis"):
                reasoning_parts.append(f"Scene: {result.scene_analysis}")
            if hasattr(result, "motion_planning"):
                reasoning_parts.append(f"Motion: {result.motion_planning}")
            if hasattr(result, "style_selection"):
                reasoning_parts.append(f"Style: {result.style_selection}")
            
            reasoning = "\n".join(reasoning_parts) if reasoning_parts else None
            
        except Exception:
            # Fall back to simple generation
            enhanced = prompt
            reasoning = None
        
        # Generate video
        video_path = self.video_generator(enhanced, **kwargs)
        
        return VideoOutput(
            video_path=video_path,
            enhanced_prompt=enhanced,
            reasoning=reasoning,
            metadata={
                "original_prompt": prompt,
                "method": "chain_of_thought",
            },
        )


class VideoReAct(VideoModule):
    """
    ReAct-style video generation with iterative refinement.
    
    Uses an observation-thought-action loop to iteratively
    refine video generation, potentially regenerating multiple
    times to improve quality.
    
    Example:
        >>> react = VideoReAct("prompt -> video", max_iterations=3)
        >>> result = react("a robot walking through a forest")
    """
    
    def __init__(
        self,
        signature: Union[str, type] = "prompt -> video",
        video_generator: Optional[Callable] = None,
        max_iterations: int = 3,
        quality_threshold: float = 0.8,
    ) -> None:
        """
        Initialize VideoReAct.
        
        Args:
            signature: Signature string or class.
            video_generator: Callable for video generation.
            max_iterations: Maximum refinement iterations.
            quality_threshold: Stop early if quality exceeds this.
        """
        super().__init__(signature, video_generator)
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        
        # Create ReAct module
        self._react = dspy.ReAct(VideoReActSignature, max_iters=max_iterations)
    
    def forward(self, prompt: str, **kwargs: Any) -> VideoOutput:
        """Generate a video with ReAct-style refinement."""
        iterations = []
        enhanced = prompt
        video_path = None
        
        try:
            # Run ReAct loop
            result = self._react(prompt=prompt)
            
            enhanced = getattr(result, "action_input", prompt)
            
            # Collect reasoning
            if hasattr(result, "trajectory"):
                for step in result.trajectory:
                    iterations.append({
                        "observation": getattr(step, "observation", ""),
                        "thought": getattr(step, "thought", ""),
                        "action": getattr(step, "action", ""),
                    })
            
        except Exception:
            # Fall back to simple generation
            enhanced = prompt
        
        # Generate final video
        video_path = self.video_generator(enhanced, **kwargs)
        
        reasoning = json.dumps(iterations, indent=2) if iterations else None
        
        return VideoOutput(
            video_path=video_path,
            enhanced_prompt=enhanced,
            reasoning=reasoning,
            metadata={
                "original_prompt": prompt,
                "method": "react",
                "iterations": iterations,
                "num_iterations": len(iterations),
            },
        )


class VideoEnsemble(VideoModule):
    """
    Ensemble module that combines multiple video generation approaches.
    
    Generates videos using multiple modules and selects the best
    based on a quality metric.
    
    Example:
        >>> ensemble = VideoEnsemble([
        ...     VideoPredict(),
        ...     VideoChainOfThought(),
        ... ])
        >>> result = ensemble("a sunset over mountains")
    """
    
    def __init__(
        self,
        modules: List[VideoModule],
        selection_metric: Optional[Callable] = None,
    ) -> None:
        """
        Initialize VideoEnsemble.
        
        Args:
            modules: List of video modules to ensemble.
            selection_metric: Metric for selecting best output.
        """
        super().__init__()
        self.modules = modules
        self.selection_metric = selection_metric
    
    def forward(self, prompt: str, **kwargs: Any) -> VideoOutput:
        """Generate videos with all modules and select best."""
        outputs = []
        
        for module in self.modules:
            try:
                output = module(prompt, **kwargs)
                outputs.append(output)
            except Exception:
                continue
        
        if not outputs:
            raise RuntimeError("All ensemble modules failed")
        
        # If no metric, return first successful
        if self.selection_metric is None:
            return outputs[0]
        
        # Score and select best
        best_output = None
        best_score = float("-inf")
        
        for output in outputs:
            try:
                # Create a mock example for the metric
                from vidspy.core import Example
                example = Example(prompt=prompt, video_path=output.video_path)
                score = self.selection_metric(example, output)
                
                if score > best_score:
                    best_score = score
                    best_output = output
            except Exception:
                continue
        
        if best_output is None:
            return outputs[0]
        
        best_output.metadata["ensemble_score"] = best_score
        return best_output


def create_video_module(
    signature: Union[str, type],
    module_type: str = "predict",
    **kwargs: Any,
) -> VideoModule:
    """
    Factory function to create video modules.
    
    Args:
        signature: DSPy signature string or class.
        module_type: Type of module ("predict", "cot", "react", "ensemble").
        **kwargs: Additional arguments for the module.
    
    Returns:
        Configured VideoModule instance.
    
    Example:
        >>> module = create_video_module(
        ...     "prompt -> video",
        ...     module_type="cot"
        ... )
    """
    module_types = {
        "predict": VideoPredict,
        "cot": VideoChainOfThought,
        "chain_of_thought": VideoChainOfThought,
        "react": VideoReAct,
    }
    
    if module_type not in module_types:
        raise ValueError(
            f"Unknown module type: {module_type}. "
            f"Available: {list(module_types.keys())}"
        )
    
    return module_types[module_type](signature=signature, **kwargs)
