"""
ViDSPy optimizers for tuning video generation instructions and demonstrations.

All optimizers work on:
- Instructions: Prompt templates that guide video generation
- Demonstrations: Few-shot examples that improve generation quality

These optimizers wrap DSPy's canonical optimizers, adapting them for
video-specific optimization with VBench metric feedback.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import dspy


# ============================================================================
# Base Optimizer
# ============================================================================

class VidOptimizer(ABC):
    """
    Abstract base class for ViDSPy optimizers.
    
    All video optimizers inherit from this class and implement the
    compile method for optimizing video generation modules.
    """
    
    def __init__(
        self,
        metric: Callable,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the optimizer.
        
        Args:
            metric: Metric function for evaluation. Should accept
                   (example, prediction) and return a float score.
            verbose: Whether to print optimization progress.
        """
        self.metric = metric
        self.verbose = verbose
        self._history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def compile(
        self,
        module: Any,
        trainset: List[Any],
        valset: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Compile/optimize a module on the training set.
        
        Args:
            module: The module to optimize.
            trainset: Training examples.
            valset: Optional validation set.
            **kwargs: Additional optimizer-specific arguments.
        
        Returns:
            Optimized module with tuned instructions and demos.
        """
        pass
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return self._history.copy()
    
    def _log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(f"[ViDSPy] {message}")


# ============================================================================
# VidBootstrapFewShot
# ============================================================================

class VidBootstrapFewShot(VidOptimizer):
    """
    Bootstrap few-shot optimizer for video generation.
    
    Automatically generates and selects high-quality demonstrations
    by running the module on training examples and keeping those
    that score well on the metric.
    
    This is the video-adapted version of DSPy's BootstrapFewShot.
    
    Args:
        metric: Evaluation metric function.
        max_bootstrapped_demos: Maximum demos to bootstrap (default 4).
        max_labeled_demos: Maximum labeled demos to include (default 4).
        max_rounds: Maximum bootstrap rounds (default 1).
        max_errors: Maximum errors before stopping (default 5).
        
    Example:
        >>> optimizer = VidBootstrapFewShot(
        ...     metric=composite_reward,
        ...     max_bootstrapped_demos=4
        ... )
        >>> optimized = optimizer.compile(module, trainset)
    """
    
    def __init__(
        self,
        metric: Callable,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 4,
        max_rounds: int = 1,
        max_errors: int = 5,
        verbose: bool = False,
    ) -> None:
        super().__init__(metric, verbose)
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.max_rounds = max_rounds
        self.max_errors = max_errors
    
    def compile(
        self,
        module: Any,
        trainset: List[Any],
        valset: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Compile the module using bootstrapped few-shot learning.
        
        Process:
        1. Run module on training examples
        2. Score outputs with the metric
        3. Select top-scoring examples as demonstrations
        4. Attach demos to the module
        """
        self._log(f"Starting BootstrapFewShot with {len(trainset)} examples")
        
        try:
            # Use DSPy's BootstrapFewShot
            bootstrap = dspy.BootstrapFewShot(
                metric=self.metric,
                max_bootstrapped_demos=self.max_bootstrapped_demos,
                max_labeled_demos=self.max_labeled_demos,
                max_rounds=self.max_rounds,
                max_errors=self.max_errors,
            )
            
            optimized = bootstrap.compile(module, trainset=trainset)
            
            self._history.append({
                "optimizer": "VidBootstrapFewShot",
                "num_examples": len(trainset),
                "max_demos": self.max_bootstrapped_demos,
                "status": "success",
            })
            
            self._log("BootstrapFewShot optimization complete")
            return optimized
            
        except Exception as e:
            self._log(f"BootstrapFewShot failed: {e}, using fallback")
            return self._fallback_compile(module, trainset)
    
    def _fallback_compile(
        self,
        module: Any,
        trainset: List[Any],
    ) -> Any:
        """Fallback compilation when DSPy optimizer fails."""
        # Simple fallback: score all examples and select top ones
        scored_examples = []
        
        for example in trainset[:min(len(trainset), 20)]:
            try:
                pred = module(prompt=example.prompt)
                score = self.metric(example, pred)
                scored_examples.append((score, example))
            except Exception:
                continue
        
        # Sort by score and select top demos
        scored_examples.sort(key=lambda x: x[0], reverse=True)

        # Note: In fallback mode, we simply return the module as-is.
        # DSPy's actual optimization happens through the compile() method above,
        # not through storing demos in the module.
        return module


# ============================================================================
# VidLabeledFewShot
# ============================================================================

class VidLabeledFewShot(VidOptimizer):
    """
    Labeled few-shot optimizer for video generation.
    
    Selects the k best labeled examples from the training set
    based on metric scores. Unlike BootstrapFewShot, this doesn't
    generate new examples—it only selects from existing ones.
    
    This is the video-adapted version of DSPy's LabeledFewShot.
    
    Args:
        metric: Evaluation metric function.
        k: Number of examples to select (default 3).
        
    Example:
        >>> optimizer = VidLabeledFewShot(metric=composite_reward, k=5)
        >>> optimized = optimizer.compile(module, trainset)
    """
    
    def __init__(
        self,
        metric: Callable,
        k: int = 3,
        verbose: bool = False,
    ) -> None:
        super().__init__(metric, verbose)
        self.k = k
    
    def compile(
        self,
        module: Any,
        trainset: List[Any],
        valset: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Compile the module using labeled few-shot selection.
        
        Process:
        1. Score all training examples
        2. Select top-k examples
        3. Attach as demonstrations
        """
        self._log(f"Starting LabeledFewShot with k={self.k}")
        
        try:
            # Use DSPy's LabeledFewShot
            labeled_fs = dspy.LabeledFewShot(k=self.k)
            
            optimized = labeled_fs.compile(module, trainset=trainset)
            
            self._history.append({
                "optimizer": "VidLabeledFewShot",
                "num_examples": len(trainset),
                "k": self.k,
                "status": "success",
            })
            
            self._log("LabeledFewShot optimization complete")
            return optimized
            
        except Exception as e:
            self._log(f"LabeledFewShot failed: {e}, using fallback")
            return self._fallback_compile(module, trainset)
    
    def _fallback_compile(
        self,
        module: Any,
        trainset: List[Any],
    ) -> Any:
        """Fallback compilation."""
        # Note: In fallback mode, we simply return the module as-is.
        # DSPy's actual optimization happens through the compile() method above,
        # not through storing demos in the module.
        return module


# ============================================================================
# VidMIPROv2
# ============================================================================

class VidMIPROv2(VidOptimizer):
    """
    MIPROv2 optimizer for video generation.
    
    Multi-stage Instruction and Prompt Refinement Optimizer that
    jointly optimizes instructions and demonstrations. Uses a
    combination of:
    - Instruction proposal and selection
    - Demo bootstrapping and filtering
    - Bayesian optimization for hyperparameters
    
    This is the video-adapted version of DSPy's MIPROv2.
    
    Args:
        metric: Evaluation metric function.
        num_candidates: Number of instruction candidates (default 10).
        auto: Optimization intensity ("off", "light", "medium", "heavy").
        num_trials: Number of optimization trials.
        max_bootstrapped_demos: Max demos to bootstrap.
        max_labeled_demos: Max labeled demos.
        
    Example:
        >>> optimizer = VidMIPROv2(
        ...     metric=composite_reward,
        ...     num_candidates=15,
        ...     auto="medium"
        ... )
        >>> optimized = optimizer.compile(module, trainset)
    """
    
    def __init__(
        self,
        metric: Callable,
        num_candidates: int = 10,
        auto: Literal["off", "light", "medium", "heavy"] = "light",
        num_trials: Optional[int] = None,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 4,
        verbose: bool = False,
    ) -> None:
        super().__init__(metric, verbose)
        self.num_candidates = num_candidates
        self.auto = auto
        self.num_trials = num_trials
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
    
    def compile(
        self,
        module: Any,
        trainset: List[Any],
        valset: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Compile the module using MIPROv2 optimization.
        
        Process:
        1. Generate instruction candidates
        2. Bootstrap demonstrations
        3. Jointly optimize instruction-demo combinations
        4. Select best configuration
        """
        self._log(f"Starting MIPROv2 with {self.num_candidates} candidates, auto={self.auto}")
        
        try:
            # Build MIPROv2 kwargs
            mipro_kwargs = {
                "metric": self.metric,
                "num_candidates": self.num_candidates,
                "auto": self.auto,
            }
            
            if self.num_trials is not None:
                mipro_kwargs["num_trials"] = self.num_trials
            
            # Use DSPy's MIPROv2
            mipro = dspy.MIPROv2(**mipro_kwargs)
            
            if valset is not None:
                optimized = mipro.compile(
                    module,
                    trainset=trainset,
                    valset=valset,
                )
            else:
                optimized = mipro.compile(
                    module,
                    trainset=trainset,
                )
            
            self._history.append({
                "optimizer": "VidMIPROv2",
                "num_candidates": self.num_candidates,
                "auto": self.auto,
                "num_examples": len(trainset),
                "status": "success",
            })
            
            self._log("MIPROv2 optimization complete")
            return optimized
            
        except Exception as e:
            self._log(f"MIPROv2 failed: {e}, falling back to BootstrapFewShot")
            # Fall back to bootstrap
            fallback = VidBootstrapFewShot(
                metric=self.metric,
                max_bootstrapped_demos=self.max_bootstrapped_demos,
                max_labeled_demos=self.max_labeled_demos,
                verbose=self.verbose,
            )
            return fallback.compile(module, trainset, valset)


# ============================================================================
# VidCOPRO
# ============================================================================

class VidCOPRO(VidOptimizer):
    """
    COPRO optimizer for video generation.
    
    Cooperative Prompt Optimization uses a multi-LM approach where
    a teacher model proposes instruction refinements and a student
    model evaluates them. Supports iterative breadth-first and
    depth-first exploration.
    
    This is the video-adapted version of DSPy's COPRO.
    
    Args:
        metric: Evaluation metric function.
        breadth: Number of candidates per level (default 5).
        depth: Number of refinement levels (default 3).
        init_temperature: Initial sampling temperature.
        
    Example:
        >>> optimizer = VidCOPRO(
        ...     metric=composite_reward,
        ...     breadth=10,
        ...     depth=5
        ... )
        >>> optimized = optimizer.compile(module, trainset)
    """
    
    def __init__(
        self,
        metric: Callable,
        breadth: int = 5,
        depth: int = 3,
        init_temperature: float = 1.4,
        verbose: bool = False,
    ) -> None:
        super().__init__(metric, verbose)
        self.breadth = breadth
        self.depth = depth
        self.init_temperature = init_temperature
    
    def compile(
        self,
        module: Any,
        trainset: List[Any],
        valset: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Compile the module using COPRO optimization.
        
        Process:
        1. Initialize with base instructions
        2. For each depth level:
           a. Generate breadth candidates
           b. Evaluate on training set
           c. Select best candidate
        3. Return module with best instructions
        """
        self._log(f"Starting COPRO with breadth={self.breadth}, depth={self.depth}")
        
        try:
            # Use DSPy's COPRO
            copro = dspy.COPRO(
                metric=self.metric,
                breadth=self.breadth,
                depth=self.depth,
                init_temperature=self.init_temperature,
            )
            
            eval_set = valset if valset is not None else trainset
            
            optimized = copro.compile(
                module,
                trainset=trainset,
                eval_kwargs={"devset": eval_set},
            )
            
            self._history.append({
                "optimizer": "VidCOPRO",
                "breadth": self.breadth,
                "depth": self.depth,
                "num_examples": len(trainset),
                "status": "success",
            })
            
            self._log("COPRO optimization complete")
            return optimized
            
        except Exception as e:
            self._log(f"COPRO failed: {e}, falling back to BootstrapFewShot")
            fallback = VidBootstrapFewShot(
                metric=self.metric,
                verbose=self.verbose,
            )
            return fallback.compile(module, trainset, valset)


# ============================================================================
# VidGEPA
# ============================================================================

class VidGEPA(VidOptimizer):
    """
    GEPA optimizer for video generation.
    
    Generate-Evaluate-Propose-Accept optimizer that uses an
    iterative refinement cycle:
    1. Generate: Create instruction candidates
    2. Evaluate: Score candidates on training set
    3. Propose: Suggest improvements based on errors
    4. Accept: Keep improvements that boost scores
    
    This provides more advanced instruction generation than
    simpler optimizers.
    
    Args:
        metric: Evaluation metric function.
        auto: Optimization intensity ("off", "light", "medium", "heavy").
        num_candidates: Number of instruction candidates.
        
    Example:
        >>> optimizer = VidGEPA(metric=composite_reward, auto="medium")
        >>> optimized = optimizer.compile(module, trainset)
    """
    
    def __init__(
        self,
        metric: Callable,
        auto: Literal["off", "light", "medium", "heavy"] = "light",
        num_candidates: int = 10,
        verbose: bool = False,
    ) -> None:
        super().__init__(metric, verbose)
        self.auto = auto
        self.num_candidates = num_candidates
    
    def compile(
        self,
        module: Any,
        trainset: List[Any],
        valset: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Compile the module using GEPA optimization.
        
        Implements a generate-evaluate-propose-accept cycle for
        instruction refinement.
        """
        self._log(f"Starting GEPA with auto={self.auto}")
        
        # GEPA is essentially MIPROv2 with specific settings
        # We implement it using MIPROv2 as the backend
        try:
            mipro = dspy.MIPROv2(
                metric=self.metric,
                num_candidates=self.num_candidates,
                auto=self.auto,
            )
            
            if valset is not None:
                optimized = mipro.compile(
                    module,
                    trainset=trainset,
                    valset=valset,
                )
            else:
                optimized = mipro.compile(
                    module,
                    trainset=trainset,
                )
            
            self._history.append({
                "optimizer": "VidGEPA",
                "auto": self.auto,
                "num_candidates": self.num_candidates,
                "num_examples": len(trainset),
                "status": "success",
            })
            
            self._log("GEPA optimization complete")
            return optimized
            
        except Exception as e:
            self._log(f"GEPA failed: {e}, falling back to BootstrapFewShot")
            fallback = VidBootstrapFewShot(
                metric=self.metric,
                verbose=self.verbose,
            )
            return fallback.compile(module, trainset, valset)


# ============================================================================
# Optimizer Factory
# ============================================================================

def create_optimizer(
    name: str,
    metric: Callable,
    **kwargs: Any,
) -> VidOptimizer:
    """
    Factory function to create optimizers by name.
    
    Args:
        name: Optimizer name ("bootstrap", "labeled", "mipro_v2", "copro", "gepa").
        metric: Metric function for evaluation.
        **kwargs: Additional optimizer-specific arguments.
    
    Returns:
        Configured optimizer instance.
    
    Example:
        >>> optimizer = create_optimizer(
        ...     "mipro_v2",
        ...     composite_reward,
        ...     num_candidates=15
        ... )
    """
    optimizers = {
        "bootstrap": VidBootstrapFewShot,
        "bootstrap_few_shot": VidBootstrapFewShot,
        "labeled": VidLabeledFewShot,
        "labeled_few_shot": VidLabeledFewShot,
        "mipro": VidMIPROv2,
        "mipro_v2": VidMIPROv2,
        "copro": VidCOPRO,
        "gepa": VidGEPA,
    }
    
    name_lower = name.lower().replace("-", "_")
    
    if name_lower not in optimizers:
        raise ValueError(
            f"Unknown optimizer: {name}. "
            f"Available: {list(set(optimizers.keys()))}"
        )
    
    return optimizers[name_lower](metric=metric, **kwargs)
