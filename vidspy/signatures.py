"""
Video-specific DSPy signatures for text-to-video generation.
"""

from __future__ import annotations

from typing import Any, List, Optional

import dspy


class VideoSignature(dspy.Signature):
    """
    Base signature for video generation tasks.
    
    This signature defines the input-output structure for text-to-video
    generation, including the prompt template that will be optimized.
    
    Attributes:
        prompt: Input text description of the desired video.
        video_path: Output path to the generated video file.
        
    The prompt_template field is optimized during training to improve
    video quality and text-video alignment scores.
    """
    
    prompt: str = dspy.InputField(
        desc="Text description of the desired video content"
    )
    video_path: str = dspy.OutputField(
        desc="Path to the generated video file"
    )


class VideoGenerationSignature(dspy.Signature):
    """
    Signature for video generation with enhanced prompt engineering.
    
    This signature includes additional fields for structured prompt
    generation that leads to higher-quality videos.
    """
    
    prompt: str = dspy.InputField(
        desc="Original text description of the desired video"
    )
    enhanced_prompt: str = dspy.OutputField(
        desc="Enhanced and detailed prompt for video generation"
    )
    style_hints: str = dspy.OutputField(
        desc="Visual style and aesthetic hints for the video"
    )
    motion_description: str = dspy.OutputField(
        desc="Description of motion and temporal dynamics"
    )
    video_path: str = dspy.OutputField(
        desc="Path to the generated video file"
    )


class VideoQualitySignature(dspy.Signature):
    """
    Signature for video quality assessment.
    
    Used by VLMs to evaluate video quality metrics including
    subject consistency, motion smoothness, and aesthetic quality.
    """
    
    video_path: str = dspy.InputField(
        desc="Path to the video file to evaluate"
    )
    subject_consistency_score: float = dspy.OutputField(
        desc="Score for temporal stability of subjects (0-1)"
    )
    motion_smoothness_score: float = dspy.OutputField(
        desc="Score for natural motion quality (0-1)"
    )
    flickering_score: float = dspy.OutputField(
        desc="Score for absence of temporal jitter (0-1)"
    )
    anatomy_score: float = dspy.OutputField(
        desc="Score for correct human anatomy rendering (0-1)"
    )
    aesthetic_score: float = dspy.OutputField(
        desc="Score for artistic/visual beauty (0-1)"
    )
    imaging_quality_score: float = dspy.OutputField(
        desc="Score for technical clarity and sharpness (0-1)"
    )
    quality_reasoning: str = dspy.OutputField(
        desc="Reasoning for the quality scores"
    )


class VideoAlignmentSignature(dspy.Signature):
    """
    Signature for text-video alignment assessment.
    
    Used to evaluate how well a generated video matches its prompt
    across multiple semantic dimensions.
    """
    
    prompt: str = dspy.InputField(
        desc="Original text prompt for the video"
    )
    video_path: str = dspy.InputField(
        desc="Path to the video file to evaluate"
    )
    object_class_score: float = dspy.OutputField(
        desc="Score for correct objects appearing (0-1)"
    )
    human_action_score: float = dspy.OutputField(
        desc="Score for correct actions being performed (0-1)"
    )
    spatial_relationship_score: float = dspy.OutputField(
        desc="Score for correct spatial layout (0-1)"
    )
    overall_consistency_score: float = dspy.OutputField(
        desc="Score for holistic text-video alignment (0-1)"
    )
    alignment_reasoning: str = dspy.OutputField(
        desc="Reasoning for the alignment scores"
    )


class VideoPromptEnhancementSignature(dspy.Signature):
    """
    Signature for enhancing video generation prompts.
    
    Takes a simple prompt and produces an enhanced version with
    detailed descriptions for better video generation results.
    """
    
    original_prompt: str = dspy.InputField(
        desc="Original simple prompt from user"
    )
    enhanced_prompt: str = dspy.OutputField(
        desc=(
            "Enhanced prompt with detailed visual descriptions including: "
            "subject details, actions, environment, lighting, camera movement, "
            "style, and temporal progression"
        )
    )


class VideoFewShotDemoSignature(dspy.Signature):
    """
    Signature for few-shot demonstration selection.
    
    Used by optimizers to select the most relevant demonstrations
    for a given input prompt.
    """
    
    input_prompt: str = dspy.InputField(
        desc="The input prompt to find demonstrations for"
    )
    candidate_demos: str = dspy.InputField(
        desc="JSON list of candidate demonstration prompts"
    )
    selected_demo_indices: str = dspy.OutputField(
        desc="Comma-separated indices of selected demonstrations"
    )
    selection_reasoning: str = dspy.OutputField(
        desc="Reasoning for why these demonstrations were selected"
    )


class VideoChainOfThoughtSignature(dspy.Signature):
    """
    Signature for chain-of-thought video generation.
    
    Includes intermediate reasoning steps for more coherent
    video generation planning.
    """
    
    prompt: str = dspy.InputField(
        desc="Text description of the desired video"
    )
    scene_analysis: str = dspy.OutputField(
        desc="Analysis of the scene elements and their relationships"
    )
    motion_planning: str = dspy.OutputField(
        desc="Plan for motion and temporal dynamics in the video"
    )
    style_selection: str = dspy.OutputField(
        desc="Selection of visual style and aesthetic approach"
    )
    enhanced_prompt: str = dspy.OutputField(
        desc="Final enhanced prompt incorporating all analysis"
    )
    video_path: str = dspy.OutputField(
        desc="Path to the generated video file"
    )


class VideoReActSignature(dspy.Signature):
    """
    Signature for ReAct-style video generation with iterative refinement.
    
    Supports observation-action cycles for quality improvement.
    """
    
    prompt: str = dspy.InputField(
        desc="Text description of the desired video"
    )
    observation: str = dspy.OutputField(
        desc="Current observation about video generation requirements"
    )
    thought: str = dspy.OutputField(
        desc="Reasoning about what to do next"
    )
    action: str = dspy.OutputField(
        desc="Action to take (generate, refine, or finalize)"
    )
    action_input: str = dspy.OutputField(
        desc="Input for the action (enhanced prompt or refinement instructions)"
    )
    video_path: str = dspy.OutputField(
        desc="Path to the generated video file"
    )


def create_video_signature(
    input_fields: List[str],
    output_fields: List[str],
    instructions: Optional[str] = None,
) -> type:
    """
    Dynamically create a video signature class.
    
    Args:
        input_fields: List of input field names.
        output_fields: List of output field names.
        instructions: Optional instructions for the signature.
    
    Returns:
        A new signature class.
    
    Example:
        >>> MySignature = create_video_signature(
        ...     input_fields=["prompt", "style"],
        ...     output_fields=["video_path", "thumbnail_path"],
        ...     instructions="Generate a video with the specified style"
        ... )
    """
    # Build field definitions
    fields = {}
    for field_name in input_fields:
        fields[field_name] = dspy.InputField(desc=f"Input: {field_name}")
    for field_name in output_fields:
        fields[field_name] = dspy.OutputField(desc=f"Output: {field_name}")
    
    # Create the class
    class_name = "DynamicVideoSignature"
    bases = (dspy.Signature,)
    
    # Add instructions if provided
    if instructions:
        fields["__doc__"] = instructions
    
    return type(class_name, bases, fields)


def parse_signature_string(signature_str: str) -> tuple[List[str], List[str]]:
    """
    Parse a signature string in DSPy format.
    
    Args:
        signature_str: String like "prompt -> video_path" or 
                      "prompt, style -> video_path, quality"
    
    Returns:
        Tuple of (input_fields, output_fields).
    
    Example:
        >>> inputs, outputs = parse_signature_string("prompt -> video")
        >>> inputs
        ['prompt']
        >>> outputs
        ['video']
    """
    if "->" not in signature_str:
        raise ValueError(
            f"Invalid signature string: {signature_str}. "
            "Expected format: 'input1, input2 -> output1, output2'"
        )
    
    parts = signature_str.split("->")
    if len(parts) != 2:
        raise ValueError(f"Invalid signature string: {signature_str}")
    
    input_str, output_str = parts
    
    input_fields = [f.strip() for f in input_str.split(",") if f.strip()]
    output_fields = [f.strip() for f in output_str.split(",") if f.strip()]
    
    if not input_fields:
        raise ValueError("Signature must have at least one input field")
    if not output_fields:
        raise ValueError("Signature must have at least one output field")
    
    return input_fields, output_fields
