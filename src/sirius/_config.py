"""Configuration constants for Sirius."""

import os
from typing import Any, Final

# =============================================================================
# Generation Defaults
# =============================================================================

DEFAULT_SEED: Final[int] = 42
DEFAULT_GUIDANCE: Final[float] = 3.5  # Low for creativity + consistency
DEFAULT_WIDTH: Final[int] = 1024
DEFAULT_HEIGHT: Final[int] = 576  # 16:9 aspect ratio
DEFAULT_STEPS: Final[int] = 28
DEFAULT_FRAME_COUNT: Final[int] = 16
DEFAULT_WORKERS: Final[int] = 4  # Parallel generation (Carmack feedback)

# =============================================================================
# Video Defaults
# =============================================================================

DEFAULT_FRAMERATE: Final[int] = 12  # Smooth but forgiving
DEFAULT_VIDEO_QUALITY: Final[int] = 23  # CRF value (lower = better)

# =============================================================================
# Model Selection
# =============================================================================

# BFL FLUX.2 Models
FLUX_DEV: Final[str] = "flux-dev"  # Development, supports guidance/steps
FLUX_KLEIN: Final[str] = "flux-2-klein-9b"  # Fast generation
FLUX_PRO: Final[str] = "flux-2-pro"  # Highest quality (FLUX.2 Pro)

# Default models for different roles
ANCHOR_MODEL: Final[str] = FLUX_PRO  # Quality for first/last frames
ANIMATOR_MODEL: Final[str] = FLUX_PRO  # Use Pro for all frames (as requested)

# Claude model
CLAUDE_MODEL: Final[str] = "claude-sonnet-4-20250514"

# =============================================================================
# Runware Models
# =============================================================================

RUNWARE_FLUX_PRO: Final[str] = "runware:101@1"  # Default Flux.1 Pro on Runware
RUNWARE_FLUX_DEV: Final[str] = "runware:100@1"
RUNWARE_FLUX_SCHNELL: Final[str] = "runware:99@1"

# =============================================================================
# API Endpoints
# =============================================================================

BFL_API_BASE: Final[str] = "https://api.bfl.ai/v1"
ANTHROPIC_API_BASE: Final[str] = "https://api.anthropic.com/v1"

# =============================================================================
# Timeouts & Retries
# =============================================================================

BFL_POLL_INTERVAL: Final[float] = 0.5  # Seconds between status checks
BFL_POLL_TIMEOUT: Final[float] = 120.0  # Max seconds to wait for generation
BFL_MAX_RETRIES: Final[int] = 3

ANTHROPIC_TIMEOUT: Final[float] = 60.0
ANTHROPIC_MAX_RETRIES: Final[int] = 2

# =============================================================================
# Environment Variables
# =============================================================================


def get_bfl_api_key() -> str:
    """Get BFL API key from environment."""
    key = os.environ.get("BFL_API_KEY")
    if not key:
        raise ValueError(
            "BFL_API_KEY environment variable not set. "
            "Get your API key at https://api.bfl.ml/"
        )
    return key


def get_anthropic_api_key() -> str:
    """Get Anthropic API key from environment."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Get your API key at https://console.anthropic.com/"
        )
    return key


def get_runware_api_key() -> str:
    """Get Runware API key from environment."""
    key = os.environ.get("RUNWARE_API_KEY")
    # Return None if not set, we'll fall back to BFL
    return key


# =============================================================================
# Transition Style Descriptions (for Director prompts)
# =============================================================================

TRANSITION_DESCRIPTIONS: Final[dict[str, str]] = {
    "morph": (
        "Create a smooth visual morph between the images. "
        "Gradually blend visual elements, colors, and shapes. "
        "Each frame should look like a natural intermediate state."
    ),
    "narrative": (
        "Create a story-driven transition. Imagine what happens between "
        "these two moments in time. Include actions, movements, or events "
        "that connect the start and end states narratively."
    ),
    "fade": (
        "Create a gradual style and subject blend. Slowly transform "
        "colors, textures, and features from the first image to the second. "
        "Focus on smooth color transitions and feature morphing."
    ),
    "surreal": (
        "Create dreamlike, artistic transitions. Take creative liberties "
        "with abstract intermediates. The middle frames can be surreal, "
        "impossible, or fantastical before resolving to the end state."
    ),
}

# =============================================================================
# FLUX Consistency Parameters
# =============================================================================

FLUX_CONSISTENCY_PARAMS: Final[dict[str, Any]] = {
    # ALWAYS LOCK THESE (same value every frame)
    "locked": {
        "aspect_ratio": "16:9",  # or "1:1", "9:16" — pick one, never change
        "resolution": "1024x576",  # match aspect ratio, Flux sweet spot
        "seed": 42,  # same seed = same noise = more consistency
        "guidance_scale": 3.5,  # Schnell works well here
    },
    # INTERPOLATE THESE (gradually shift from A to B)
    "interpolated": {
        "lighting_direction": {
            "format": "lit from the {direction}",
            "options": ["left", "right", "above", "below", "behind"],
            "rule": "Only change if A and B have DIFFERENT lighting. Shift gradually.",
        },
        "lighting_quality": {
            "format": "{quality} lighting",
            "options": [
                "soft diffused",
                "harsh direct",
                "golden hour",
                "overcast",
                "dramatic chiaroscuro",
            ],
            "rule": "Interpolate through intermediate qualities, don't jump.",
        },
        "color_temperature": {
            "format": "{temp} color temperature",
            "options": ["warm amber", "neutral daylight", "cool blue"],
            "rule": "Shift linearly. Warm→Cool should pass through neutral.",
        },
        "depth_of_field": {
            "format": "{dof}",
            "options": [
                "shallow depth of field, background blurred",
                "deep focus, everything sharp",
            ],
            "rule": "Only change if narratively motivated.",
        },
    },
    # SPECIFY EXPLICITLY EVERY FRAME
    "explicit_per_frame": {
        "camera_angle": "Specify in each prompt: 'eye-level shot', 'low angle', 'overhead view'",
        "subject_position": "Specify where subject is in frame: 'centered', 'rule of thirds left'",
        "style_keywords": "Repeat in every prompt: 'photorealistic, 8k, detailed' or 'oil painting, impressionist'",
    },
}
