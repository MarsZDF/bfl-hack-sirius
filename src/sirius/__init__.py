"""Sirius: AI Image Morphing Pipeline.

Transform two images into a smooth video transition using
Claude Vision for analysis and FLUX.2 for generation.

Example:
    >>> from sirius import morph
    >>> result = morph("cat.png", "dog.png")
    >>> print(f"Video: {result.video_path}")

    >>> # With progress tracking
    >>> from sirius import morph, TransitionStyle
    >>> def on_progress(update):
    ...     print(f"{update.progress:.0%} - {update.message}")
    >>> result = morph(
    ...     "a.png", "b.png",
    ...     transition_style=TransitionStyle.NARRATIVE,
    ...     on_progress=on_progress,
    ... )
"""

__version__ = "0.1.0"

# Main API
# Types
from ._types import (
    AspectRatio,
    Frame,
    GenerationConfig,
    ImageAnalysis,
    MorphResult,
    TransformType,
    TransitionPlan,
    TransitionStyle,
    VideoConfig,
)
from .animator import animate, create_config, generate_frames_parallel

# Individual pipeline stages (for advanced usage)
from .director import (
    analyze_images,
    analyze_single_image,
    describe_changes,
    describe_images,
    direct,
    plan_transition,
    plan_transform,
    transform,
)
from .editor import add_ambient_audio, add_audio, assemble_gif, assemble_video, create_preview, edit

# Exceptions
from .exceptions import (
    AnalysisError,
    AssemblyError,
    BFLAPIError,
    CancelledException,
    ConfigurationError,
    FrameGenerationError,
    GenerationError,
    ImageLoadError,
    PipelineError,
    SiriusError,
    VideoEncodingError,
    VisionAPIError,
)
from .pipeline import morph, morph_async, morph_stream, plan_morph

# Progress & Cancellation
from .progress import (
    CancellationToken,
    MorphStage,
    ProgressCallback,
    ProgressUpdate,
)

__all__ = [
    # Version
    "__version__",
    # Main API
    "morph",
    "morph_async",
    "morph_stream",
    "plan_morph",
    # Types
    "AspectRatio",
    "Frame",
    "GenerationConfig",
    "ImageAnalysis",
    "MorphResult",
    "TransformType",
    "TransitionPlan",
    "TransitionStyle",
    "VideoConfig",
    # Progress
    "CancellationToken",
    "MorphStage",
    "ProgressCallback",
    "ProgressUpdate",
    # Exceptions
    "AnalysisError",
    "AssemblyError",
    "BFLAPIError",
    "CancelledException",
    "ConfigurationError",
    "FrameGenerationError",
    "GenerationError",
    "ImageLoadError",
    "PipelineError",
    "SiriusError",
    "VideoEncodingError",
    "VisionAPIError",
    # Pipeline stages
    "analyze_images",
    "analyze_single_image",
    "describe_changes",
    "describe_images",
    "animate",
    "assemble_gif",
    "assemble_video",
    "create_config",
    "create_preview",
    "direct",
    "edit",
    "generate_frames_parallel",
    "plan_transform",
    "plan_transition",
    "transform",
    # Audio
    "add_audio",
    "add_ambient_audio",
]
