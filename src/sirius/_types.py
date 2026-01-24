"""Data types for Sirius image morphing pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TransitionStyle(str, Enum):
    """Available transition styles for morphing."""

    MORPH = "morph"  # Direct visual interpolation
    NARRATIVE = "narrative"  # Story-driven transition
    FADE = "fade"  # Gradual style/subject blend
    SURREAL = "surreal"  # Creative dreamlike transitions


@dataclass
class ImageAnalysis:
    """Result of Claude Vision image analysis."""

    prompt: str  # Full generation prompt
    subject: str  # Main subject description
    style: str  # Art style (photorealistic, illustration, etc.)
    lighting: str  # Lighting description
    mood: str  # Emotional tone
    colors: list[str]  # Dominant colors
    summary: str  # One-line summary

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt": self.prompt,
            "subject": self.subject,
            "style": self.style,
            "lighting": self.lighting,
            "mood": self.mood,
            "colors": self.colors,
            "summary": self.summary,
        }


@dataclass
class TransitionPlan:
    """Plan for transitioning between two images."""

    prompts: list[str]  # N intermediate prompts
    frame_count: int  # Number of frames
    transition_style: TransitionStyle  # Style used
    analysis_a: ImageAnalysis  # Source image analysis
    analysis_b: ImageAnalysis  # Target image analysis

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompts": self.prompts,
            "frame_count": self.frame_count,
            "transition_style": self.transition_style.value,
            "analysis_a": self.analysis_a.to_dict(),
            "analysis_b": self.analysis_b.to_dict(),
        }


@dataclass
class GenerationConfig:
    """Configuration for FLUX.2 image generation."""

    seed: int = 42
    guidance: float = 3.5
    width: int = 1024
    height: int = 576
    steps: int = 28
    workers: int = 4  # Parallel generation workers (Carmack feedback)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "seed": self.seed,
            "guidance": self.guidance,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "workers": self.workers,
        }


@dataclass
class Frame:
    """A single generated frame in the morph sequence."""

    index: int  # Frame position (0 = first, N-1 = last)
    prompt: str  # Prompt used to generate
    path: str  # Path to saved image file
    duration_ms: int  # Generation time in milliseconds
    model: str = "flux-2-klein-9b"  # Model used
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "prompt": self.prompt,
            "path": self.path,
            "duration_ms": self.duration_ms,
            "model": self.model,
            "seed": self.seed,
        }


@dataclass
class VideoConfig:
    """Configuration for video assembly."""

    framerate: int = 12
    codec: str = "libx264"
    pixel_format: str = "yuv420p"
    quality: int = 23  # CRF value (lower = better quality)
    boomerang: bool = False  # A→B→A loop (Roa feedback)


@dataclass
class MorphResult:
    """Result of a complete morph operation."""

    video_path: str  # Path to output video
    frames: list[Frame]  # All generated frames
    plan: TransitionPlan  # The transition plan used
    duration_ms: int  # Total operation time
    experiment_ids: list[int] = field(default_factory=list)  # Gallium tracking IDs
    transition_id: str = ""  # Unique ID for this morph
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "video_path": self.video_path,
            "frames": [f.to_dict() for f in self.frames],
            "plan": self.plan.to_dict(),
            "duration_ms": self.duration_ms,
            "experiment_ids": self.experiment_ids,
            "transition_id": self.transition_id,
            "created_at": self.created_at.isoformat(),
        }
