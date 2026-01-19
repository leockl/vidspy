"""
VLM (Vision-Language Model) providers for video understanding and evaluation.

Supported providers:
1. OpenRouter (default): Cloud-based video VLMs via unified API
2. HuggingFace Local: Local video VLMs for offline usage
"""

from __future__ import annotations

import base64
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests


# ============================================================================
# Base VLM Provider
# ============================================================================

@dataclass
class VLMResponse:
    """Response from a VLM provider."""
    text: str
    model: str
    usage: Dict[str, int]
    metadata: Dict[str, Any]


class VLMProvider(ABC):
    """
    Abstract base class for VLM providers.
    
    VLM providers handle communication with vision-language models
    for video understanding tasks.
    """
    
    @abstractmethod
    def complete(
        self,
        prompt: str,
        video_path: Optional[str] = None,
        images: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> VLMResponse:
        """
        Generate a completion from the VLM.
        
        Args:
            prompt: Text prompt for the model.
            video_path: Optional path to video file.
            images: Optional list of image paths or base64 strings.
            **kwargs: Additional model-specific parameters.
        
        Returns:
            VLMResponse with the model output.
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available."""
        pass


# ============================================================================
# OpenRouter VLM Provider
# ============================================================================

class OpenRouterVLM(VLMProvider):
    """
    OpenRouter VLM provider for cloud-based video understanding.
    
    OpenRouter provides unified access to various video-capable VLMs
    including Google Gemini, GPT-4V, and others.
    
    Args:
        model: Model identifier (e.g., "google/gemini-2.5-flash").
        api_key: OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
        base_url: API base URL.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature.

    Example:
        >>> vlm = OpenRouterVLM(
        ...     model="google/gemini-2.5-flash",
        ...     api_key="sk-..."
        ... )
        >>> response = vlm.complete(
        ...     "Describe this video",
        ...     video_path="video.mp4"
        ... )
    """

    # Models known to support video input
    VIDEO_CAPABLE_MODELS = [
        "google/gemini-2.5-flash",
        "google/gemini-pro-vision",
        "google/gemini-1.5-pro",
        "google/gemini-1.5-flash",
        "anthropic/claude-3-opus",
        "anthropic/claude-3-sonnet",
        "anthropic/claude-3-haiku",
        "openai/gpt-4-vision-preview",
        "openai/gpt-4o",
    ]
    
    def __init__(
        self,
        model: str = "google/gemini-2.5-flash",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var "
                "or pass api_key parameter."
            )
    
    def complete(
        self,
        prompt: str,
        video_path: Optional[str] = None,
        images: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> VLMResponse:
        """Generate a completion using OpenRouter API."""
        messages = self._build_messages(prompt, video_path, images)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/vidspy",
            "X-Title": "ViDSPy",
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        
        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter API error: {response.status_code} - {response.text}"
            )
        
        data = response.json()
        
        return VLMResponse(
            text=data["choices"][0]["message"]["content"],
            model=data.get("model", self.model),
            usage=data.get("usage", {}),
            metadata={"provider": "openrouter"},
        )
    
    def _build_messages(
        self,
        prompt: str,
        video_path: Optional[str],
        images: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """Build the messages array for the API request."""
        content: List[Dict[str, Any]] = []
        
        # Add video frames if provided
        if video_path and os.path.exists(video_path):
            frames = self._extract_video_frames(video_path)
            for frame in frames:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame}"
                    }
                })
        
        # Add images if provided
        if images:
            for img in images:
                if os.path.exists(img):
                    with open(img, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode()
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_data}"
                        }
                    })
                elif img.startswith("data:"):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": img}
                    })
        
        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })
        
        return [{"role": "user", "content": content}]
    
    def _extract_video_frames(
        self,
        video_path: str,
        num_frames: int = 8,
    ) -> List[str]:
        """Extract frames from video as base64 strings."""
        try:
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                return []
            
            # Sample frames evenly
            frame_indices = [
                int(i * total_frames / num_frames)
                for i in range(num_frames)
            ]
            
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    # Resize for API efficiency
                    frame = cv2.resize(frame, (512, 512))
                    _, buffer = cv2.imencode(".jpg", frame)
                    frames.append(base64.b64encode(buffer).decode())
            
            cap.release()
            return frames
            
        except ImportError:
            print("Warning: opencv-python not installed, cannot extract video frames")
            return []
        except Exception as e:
            print(f"Warning: Failed to extract video frames: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if OpenRouter API is available."""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            return response.status_code == 200
        except Exception:
            return False


# ============================================================================
# HuggingFace VLM Provider
# ============================================================================

class HuggingFaceVLM(VLMProvider):
    """
    HuggingFace local VLM provider for offline video understanding.
    
    Runs video-capable VLMs locally using the transformers library.
    Supports models like LLaVA, Video-LLaMA, and others.
    
    Args:
        model: Model identifier from HuggingFace Hub.
        device: Device for inference ("cuda", "cpu", or "auto").
        cache_dir: Directory for caching model weights.
        torch_dtype: Torch dtype for model weights.
        
    Example:
        >>> vlm = HuggingFaceVLM(
        ...     model="llava-hf/llava-v1.6-mistral-7b-hf",
        ...     device="cuda"
        ... )
        >>> response = vlm.complete(
        ...     "What is happening in this video?",
        ...     video_path="video.mp4"
        ... )
    """
    
    def __init__(
        self,
        model: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        device: str = "auto",
        cache_dir: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> None:
        self.model_name = model
        self.device = device
        self.cache_dir = cache_dir
        self.torch_dtype = torch_dtype
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        self._model = None
        self._processor = None
        self._loaded = False
    
    def _load_model(self) -> None:
        """Lazy load the model and processor."""
        if self._loaded:
            return
        
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            # Determine dtype
            if self.torch_dtype == "float16":
                dtype = torch.float16
            elif self.torch_dtype == "bfloat16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
            
            # Determine device
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            
            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )
            
            # Load model
            model_kwargs = {
                "cache_dir": self.cache_dir,
                "torch_dtype": dtype,
            }
            
            if self.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            elif self.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
            else:
                model_kwargs["device_map"] = device
            
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                **model_kwargs,
            )
            
            self._loaded = True
            
        except ImportError as e:
            raise ImportError(
                f"Failed to load HuggingFace model. Make sure transformers "
                f"is installed: pip install transformers. Error: {e}"
            )
    
    def complete(
        self,
        prompt: str,
        video_path: Optional[str] = None,
        images: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> VLMResponse:
        """Generate a completion using local HuggingFace model."""
        self._load_model()
        
        try:
            import torch
            from PIL import Image
            
            # Prepare images
            pil_images = []
            
            if video_path and os.path.exists(video_path):
                pil_images.extend(self._extract_video_frames_pil(video_path))
            
            if images:
                for img in images:
                    if os.path.exists(img):
                        pil_images.append(Image.open(img))
            
            # Build prompt with image placeholders
            if pil_images:
                image_tokens = "<image>" * len(pil_images)
                full_prompt = f"{image_tokens}\n{prompt}"
            else:
                full_prompt = prompt
            
            # Process inputs
            inputs = self._processor(
                text=full_prompt,
                images=pil_images if pil_images else None,
                return_tensors="pt",
            )
            
            # Move to device
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get("max_tokens", 512),
                    do_sample=kwargs.get("do_sample", True),
                    temperature=kwargs.get("temperature", 0.7),
                )
            
            # Decode
            generated = self._processor.batch_decode(
                outputs,
                skip_special_tokens=True,
            )[0]
            
            # Remove the prompt from output
            if full_prompt in generated:
                generated = generated.replace(full_prompt, "").strip()
            
            return VLMResponse(
                text=generated,
                model=self.model_name,
                usage={"prompt_tokens": len(inputs.get("input_ids", [[]])[0])},
                metadata={"provider": "huggingface", "device": str(self._model.device)},
            )
            
        except Exception as e:
            raise RuntimeError(f"HuggingFace inference failed: {e}")
    
    def _extract_video_frames_pil(
        self,
        video_path: str,
        num_frames: int = 8,
    ) -> List[Any]:
        """Extract frames from video as PIL Images."""
        try:
            import cv2
            from PIL import Image
            
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                return []
            
            frame_indices = [
                int(i * total_frames / num_frames)
                for i in range(num_frames)
            ]
            
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame))
            
            cap.release()
            return frames
            
        except ImportError:
            print("Warning: opencv-python not installed")
            return []
        except Exception as e:
            print(f"Warning: Failed to extract video frames: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if the model can be loaded."""
        try:
            self._load_model()
            return True
        except Exception:
            return False


# ============================================================================
# Configuration Helpers
# ============================================================================

def configure_vlm(
    backend: str = "openrouter",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> VLMProvider:
    """
    Configure and return a VLM provider.
    
    Args:
        backend: Provider backend ("openrouter" or "huggingface").
        model: Model identifier.
        api_key: API key (for OpenRouter).
        **kwargs: Additional provider-specific arguments.
    
    Returns:
        Configured VLM provider instance.
    
    Example:
        >>> vlm = configure_vlm(
        ...     backend="openrouter",
        ...     model="google/gemini-2.5-flash"
        ... )
    """
    if backend == "openrouter":
        return OpenRouterVLM(
            model=model or "google/gemini-2.5-flash",
            api_key=api_key,
            **kwargs,
        )
    elif backend == "huggingface":
        return HuggingFaceVLM(
            model=model or "llava-hf/llava-v1.6-mistral-7b-hf",
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def list_available_models(backend: str = "openrouter") -> List[str]:
    """
    List available models for a backend.
    
    Args:
        backend: Provider backend.
    
    Returns:
        List of model identifiers.
    """
    if backend == "openrouter":
        return OpenRouterVLM.VIDEO_CAPABLE_MODELS
    elif backend == "huggingface":
        return [
            "llava-hf/llava-v1.6-mistral-7b-hf",
            "llava-hf/llava-v1.6-vicuna-7b-hf",
            "llava-hf/llava-v1.6-vicuna-13b-hf",
            "Qwen/Qwen-VL-Chat",
            "microsoft/phi-3-vision-128k-instruct",
        ]
    else:
        raise ValueError(f"Unknown backend: {backend}")
