"""Progress reporting and cancellation for Codex UX.

Per Soumith Chintala feedback: Use async generators for progress.
Per Patrick Collison feedback: Clear, predictable status updates.
"""

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class MorphStage(str, Enum):
    """Pipeline stages for progress tracking."""

    INITIALIZING = "initializing"  # Setup and validation
    ANALYZING = "analyzing"  # Claude Vision analyzing images
    PLANNING = "planning"  # Screenwriter generating prompts
    GENERATING = "generating"  # FLUX.2 generating frames
    ASSEMBLING = "assembling"  # Video encoding
    COMPLETE = "complete"

    def __str__(self) -> str:
        return self.value


@dataclass
class ProgressUpdate:
    """Progress update for Codex integration."""

    stage: MorphStage
    progress: float  # 0.0 to 1.0
    message: str  # Human-readable status
    current_frame: int | None = None  # For GENERATING stage
    total_frames: int | None = None
    elapsed_ms: int = 0
    eta_ms: int | None = None  # Estimated time remaining

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization."""
        return {
            "stage": self.stage.value,
            "progress": self.progress,
            "message": self.message,
            "current_frame": self.current_frame,
            "total_frames": self.total_frames,
            "elapsed_ms": self.elapsed_ms,
            "eta_ms": self.eta_ms,
        }

    def __str__(self) -> str:
        pct = f"{self.progress:.0%}"
        if self.current_frame is not None:
            return f"[{self.stage}] {pct} - {self.message} ({self.current_frame}/{self.total_frames})"
        return f"[{self.stage}] {pct} - {self.message}"


# Type alias for progress callback
ProgressCallback = Callable[[ProgressUpdate], None]


class CancellationToken:
    """Token for cancelling long-running operations."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation."""
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancelled

    def reset(self) -> None:
        """Reset cancellation state for reuse."""
        self._cancelled = False


@dataclass
class ProgressReporter:
    """Internal class for tracking and reporting progress.

    Manages stage transitions, progress calculations, and callbacks.
    """

    total_frames: int = 16
    callback: ProgressCallback | None = None
    cancellation: CancellationToken | None = None
    _start_time: float = field(default_factory=time.time)
    _stage: MorphStage = MorphStage.INITIALIZING
    _frames_completed: int = 0
    _updates: list[ProgressUpdate] = field(default_factory=list)

    def _get_elapsed_ms(self) -> int:
        """Get elapsed time in milliseconds."""
        return int((time.time() - self._start_time) * 1000)

    def _estimate_eta(self, progress: float) -> int | None:
        """Estimate remaining time based on current progress."""
        if progress <= 0:
            return None
        elapsed = self._get_elapsed_ms()
        total_estimated = elapsed / progress
        return int(total_estimated - elapsed)

    def check_cancelled(self) -> None:
        """Check if cancelled and raise if so.

        Raises:
            CancelledException: If cancellation was requested.
        """
        if self.cancellation and self.cancellation.is_cancelled:
            from .exceptions import CancelledException

            raise CancelledException(
                stage=self._stage.value,
                progress=self._calculate_overall_progress(),
            )

    def _calculate_overall_progress(self) -> float:
        """Calculate overall pipeline progress (0-1)."""
        # Stage weights (rough time distribution)
        stage_weights = {
            MorphStage.INITIALIZING: 0.0,
            MorphStage.ANALYZING: 0.1,
            MorphStage.PLANNING: 0.15,
            MorphStage.GENERATING: 0.9,
            MorphStage.ASSEMBLING: 0.98,
            MorphStage.COMPLETE: 1.0,
        }

        base = stage_weights.get(self._stage, 0)

        # Add frame progress within generation stage
        if self._stage == MorphStage.GENERATING and self.total_frames > 0:
            frame_progress = self._frames_completed / self.total_frames
            stage_range = 0.9 - 0.15  # GENERATING covers 15% to 90%
            return 0.15 + (frame_progress * stage_range)

        return base

    def update(
        self,
        stage: MorphStage,
        message: str,
        current_frame: int | None = None,
    ) -> ProgressUpdate:
        """Report progress update.

        Args:
            stage: Current pipeline stage.
            message: Human-readable status message.
            current_frame: Current frame being processed (for GENERATING).

        Returns:
            The ProgressUpdate that was created.
        """
        self.check_cancelled()

        self._stage = stage
        if current_frame is not None:
            self._frames_completed = current_frame

        progress = self._calculate_overall_progress()
        elapsed = self._get_elapsed_ms()
        eta = self._estimate_eta(progress)

        update = ProgressUpdate(
            stage=stage,
            progress=progress,
            message=message,
            current_frame=current_frame,
            total_frames=self.total_frames if current_frame is not None else None,
            elapsed_ms=elapsed,
            eta_ms=eta,
        )

        self._updates.append(update)

        if self.callback:
            self.callback(update)

        return update

    def start_stage(self, stage: MorphStage, message: str | None = None) -> ProgressUpdate:
        """Start a new pipeline stage.

        Args:
            stage: The stage being started.
            message: Optional custom message.

        Returns:
            The ProgressUpdate for this stage start.
        """
        default_messages = {
            MorphStage.INITIALIZING: "Initializing pipeline...",
            MorphStage.ANALYZING: "Analyzing images with Claude Vision...",
            MorphStage.PLANNING: "Planning transition prompts...",
            MorphStage.GENERATING: "Generating frames with FLUX.2...",
            MorphStage.ASSEMBLING: "Assembling video...",
            MorphStage.COMPLETE: "Morph complete!",
        }

        msg = message or default_messages.get(stage, f"Stage: {stage}")
        return self.update(stage, msg)

    def update_frame(self, current: int, total: int, message: str | None = None) -> ProgressUpdate:
        """Update frame generation progress.

        Args:
            current: Current frame number (1-indexed).
            total: Total number of frames.
            message: Optional custom message.

        Returns:
            The ProgressUpdate.
        """
        self.total_frames = total
        msg = message or f"Generated frame {current}/{total}"
        return self.update(MorphStage.GENERATING, msg, current_frame=current)

    def complete(self, message: str = "Morph complete!") -> ProgressUpdate:
        """Mark pipeline as complete.

        Args:
            message: Completion message.

        Returns:
            Final ProgressUpdate.
        """
        return self.update(MorphStage.COMPLETE, message)

    def get_updates(self) -> list[ProgressUpdate]:
        """Get all progress updates."""
        return self._updates.copy()


def create_reporter(
    total_frames: int = 16,
    callback: ProgressCallback | None = None,
    cancellation: CancellationToken | None = None,
) -> ProgressReporter:
    """Create a progress reporter.

    Args:
        total_frames: Expected number of frames.
        callback: Optional callback for progress updates.
        cancellation: Optional cancellation token.

    Returns:
        Configured ProgressReporter.
    """
    return ProgressReporter(
        total_frames=total_frames,
        callback=callback,
        cancellation=cancellation,
    )


# =============================================================================
# Async Generator API (per Soumith Chintala feedback)
# =============================================================================


async def progress_stream(
    reporter: ProgressReporter,
    poll_interval: float = 0.1,
) -> AsyncIterator[ProgressUpdate]:
    """Async generator that yields progress updates.

    This allows for clean async iteration:
        async for update in progress_stream(reporter):
            print(update)

    Args:
        reporter: The progress reporter to stream from.
        poll_interval: Seconds between checks.

    Yields:
        ProgressUpdate objects as they're created.
    """
    import asyncio

    last_count = 0

    while True:
        updates = reporter.get_updates()
        new_updates = updates[last_count:]
        last_count = len(updates)

        for update in new_updates:
            yield update

            if update.stage == MorphStage.COMPLETE:
                return

        await asyncio.sleep(poll_interval)
