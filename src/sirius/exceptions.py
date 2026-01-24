"""Exception hierarchy for Sirius (per Patrick Collison feedback).

Clear, predictable failure modes with context for debugging and retry logic.
"""

from typing import Any


class SiriusError(Exception):
    """Base exception for all Sirius errors."""

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            ctx = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.args[0]} ({ctx})"
        return str(self.args[0])


# =============================================================================
# Analysis Errors (Step 1: Director)
# =============================================================================


class AnalysisError(SiriusError):
    """Raised when image analysis fails."""

    pass


class ImageLoadError(AnalysisError):
    """Raised when an input image cannot be loaded."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(
            f"Failed to load image: {reason}",
            context={"path": path, "reason": reason},
        )
        self.path = path


class VisionAPIError(AnalysisError):
    """Raised when Claude Vision API fails."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(
            message,
            context={"status_code": status_code, "retryable": retryable},
        )
        self.status_code = status_code
        self.retryable = retryable


class AnalysisParseError(AnalysisError):
    """Raised when analysis response cannot be parsed."""

    def __init__(self, raw_response: str, reason: str) -> None:
        super().__init__(
            f"Failed to parse analysis response: {reason}",
            context={"raw_response_preview": raw_response[:200]},
        )


# =============================================================================
# Generation Errors (Step 2: Animator)
# =============================================================================


class GenerationError(SiriusError):
    """Raised when image generation fails."""

    pass


class BFLAPIError(GenerationError):
    """Raised when BFL FLUX.2 API fails."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        endpoint: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(
            message,
            context={
                "status_code": status_code,
                "endpoint": endpoint,
                "retryable": retryable,
            },
        )
        self.status_code = status_code
        self.endpoint = endpoint
        self.retryable = retryable


class FrameGenerationError(GenerationError):
    """Raised when a specific frame fails to generate."""

    def __init__(
        self,
        frame_index: int,
        prompt: str,
        reason: str,
        retryable: bool = False,
    ) -> None:
        super().__init__(
            f"Frame {frame_index} generation failed: {reason}",
            context={
                "frame_index": frame_index,
                "prompt_preview": prompt[:100],
                "retryable": retryable,
            },
        )
        self.frame_index = frame_index
        self.prompt = prompt
        self.retryable = retryable


class GenerationTimeoutError(GenerationError):
    """Raised when generation times out waiting for result."""

    def __init__(self, timeout_seconds: float, polling_url: str | None = None) -> None:
        super().__init__(
            f"Generation timed out after {timeout_seconds}s",
            context={"timeout_seconds": timeout_seconds, "polling_url": polling_url},
        )
        self.timeout_seconds = timeout_seconds


# =============================================================================
# Assembly Errors (Step 3: Editor)
# =============================================================================


class AssemblyError(SiriusError):
    """Raised when video assembly fails."""

    pass


class FrameOrderError(AssemblyError):
    """Raised when frames are missing or out of order."""

    def __init__(self, expected: int, actual: int, missing: list[int] | None = None) -> None:
        super().__init__(
            f"Frame count mismatch: expected {expected}, got {actual}",
            context={"expected": expected, "actual": actual, "missing": missing},
        )


class VideoEncodingError(AssemblyError):
    """Raised when video encoding fails."""

    def __init__(self, reason: str, output_path: str | None = None) -> None:
        super().__init__(
            f"Video encoding failed: {reason}",
            context={"output_path": output_path},
        )


# =============================================================================
# Pipeline Errors
# =============================================================================


class PipelineError(SiriusError):
    """Raised when pipeline orchestration fails."""

    pass


class CancelledException(PipelineError):
    """Raised when operation is cancelled via CancellationToken."""

    def __init__(self, stage: str, progress: float) -> None:
        super().__init__(
            f"Operation cancelled at {stage} ({progress:.0%} complete)",
            context={"stage": stage, "progress": progress},
        )
        self.stage = stage
        self.progress = progress


class ConfigurationError(PipelineError):
    """Raised when configuration is invalid."""

    def __init__(self, param: str, value: Any, reason: str) -> None:
        super().__init__(
            f"Invalid configuration: {param}={value} - {reason}",
            context={"param": param, "value": value},
        )


# =============================================================================
# Retry Helpers
# =============================================================================


def is_retryable(error: Exception) -> bool:
    """Check if an error is retryable.

    Args:
        error: The exception to check.

    Returns:
        True if the error can be retried.
    """
    if isinstance(error, (VisionAPIError, BFLAPIError, FrameGenerationError)):
        return error.retryable
    if isinstance(error, GenerationTimeoutError):
        return True  # Timeouts are usually transient
    return False


def get_retry_delay(error: Exception, attempt: int) -> float:
    """Get recommended retry delay for an error.

    Args:
        error: The exception that occurred.
        attempt: The current attempt number (1-indexed).

    Returns:
        Recommended delay in seconds before retrying.
    """
    # Exponential backoff with jitter
    base_delay = 1.0
    max_delay = 30.0

    if isinstance(error, GenerationTimeoutError):
        # Longer delay for timeouts
        base_delay = 5.0

    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
    return delay
