"""Pipeline orchestration: ties Director, Animator, and Editor together.

Main entry point for the Sirius morph operation.
"""

import time
import uuid
from pathlib import Path
from typing import Any

import gallium

from ._anthropic_client import ClaudeClient
from ._bfl_client import BFLClient
from ._config import DEFAULT_FRAME_COUNT, DEFAULT_SEED
from ._types import (
    Frame,
    GenerationConfig,
    MorphResult,
    TransitionPlan,
    TransitionStyle,
    VideoConfig,
)
from .animator import animate, create_config
from .director import direct
from .editor import edit
from .exceptions import PipelineError
from .progress import (
    CancellationToken,
    MorphStage,
    ProgressCallback,
    ProgressReporter,
    create_reporter,
)


def morph(
    image_a: str,
    image_b: str,
    *,
    frame_count: int = DEFAULT_FRAME_COUNT,
    transition_style: TransitionStyle | str = TransitionStyle.MORPH,
    seed: int = DEFAULT_SEED,
    output_dir: str = "outputs",
    video_name: str | None = None,
    boomerang: bool = False,
    on_progress: ProgressCallback | None = None,
    cancellation: CancellationToken | None = None,
    config: GenerationConfig | None = None,
    track_with_gallium: bool = True,
) -> MorphResult:
    """Morph two images into a video transition.

    This is the main entry point for Sirius. It orchestrates the full pipeline:
    1. Director: Analyze images and plan transition prompts
    2. Animator: Generate frames in parallel with FLUX.2
    3. Editor: Assemble video with imageio

    Args:
        image_a: Path to source image.
        image_b: Path to target image.
        frame_count: Number of frames to generate (default: 16).
        transition_style: Style of transition ("morph", "narrative", "fade", "surreal").
        seed: Random seed for reproducibility.
        output_dir: Directory for output files.
        video_name: Custom video filename (auto-generated if None).
        boomerang: Create A→B→A loop video.
        on_progress: Callback for progress updates (Codex integration).
        cancellation: Token for cancelling the operation.
        config: Custom generation configuration.
        track_with_gallium: Log frames to Gallium for experiment tracking.

    Returns:
        MorphResult with video path and all generated data.

    Raises:
        PipelineError: If pipeline fails at any stage.
        CancelledException: If cancelled via cancellation token.

    Example:
        >>> from sirius import morph
        >>> result = morph("cat.png", "dog.png", frame_count=16)
        >>> print(f"Video: {result.video_path}")

        >>> # With progress tracking
        >>> def on_progress(update):
        ...     print(f"{update.progress:.0%} - {update.message}")
        >>> result = morph("a.png", "b.png", on_progress=on_progress)
    """
    start_time = time.time()
    transition_id = uuid.uuid4().hex[:8]

    # Setup progress reporter
    reporter = create_reporter(
        total_frames=frame_count,
        callback=on_progress,
        cancellation=cancellation,
    )

    # Setup output directories
    base_output = Path(output_dir)
    frames_dir = base_output / "frames" / transition_id
    videos_dir = base_output / "videos"
    frames_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Create clients (reused across stages)
    claude_client = ClaudeClient()
    bfl_client = BFLClient()

    # Generation config
    if config is None:
        config = create_config(seed=seed)
    else:
        # Ensure seed is set
        config.seed = seed

    try:
        # =================================================================
        # STEP 1: DIRECTOR - Analyze images and plan transition
        # =================================================================
        reporter.start_stage(MorphStage.ANALYZING)

        plan = direct(
            image_a_path=image_a,
            image_b_path=image_b,
            frame_count=frame_count,
            style=transition_style,
            client=claude_client,
        )

        reporter.start_stage(MorphStage.PLANNING, "Transition plan created")

        # =================================================================
        # STEP 2: ANIMATOR - Generate frames in parallel
        # =================================================================
        reporter.start_stage(MorphStage.GENERATING)

        def animator_progress(current: int, total: int, message: str) -> None:
            reporter.update_frame(current, total, message)

        frames = animate(
            plan=plan,
            config=config,
            output_dir=str(frames_dir),
            use_anchors=True,
            client=bfl_client,
            on_progress=animator_progress,
        )

        # =================================================================
        # STEP 3: EDITOR - Assemble video
        # =================================================================
        reporter.start_stage(MorphStage.ASSEMBLING)

        video_filename = video_name or f"morph_{transition_id}.mp4"
        video_path = str(videos_dir / video_filename)

        video_config = VideoConfig(boomerang=boomerang)
        edit(frames, video_path, video_config)

        # =================================================================
        # TRACKING - Log to Gallium
        # =================================================================
        experiment_ids: list[int] = []
        if track_with_gallium:
            for frame in frames:
                exp_id = gallium.log(
                    prompt=frame.prompt,
                    seed=frame.seed,
                    path=frame.path,
                    model=frame.model,
                    width=config.width,
                    height=config.height,
                    duration_ms=frame.duration_ms,
                    params={
                        "guidance": config.guidance,
                        "frame_index": frame.index,
                        "transition_id": transition_id,
                        "transition_style": (
                            transition_style.value
                            if isinstance(transition_style, TransitionStyle)
                            else transition_style
                        ),
                    },
                )
                experiment_ids.append(exp_id)

        # =================================================================
        # COMPLETE
        # =================================================================
        duration_ms = int((time.time() - start_time) * 1000)
        reporter.complete(f"Video created: {video_path}")

        return MorphResult(
            video_path=video_path,
            frames=frames,
            plan=plan,
            duration_ms=duration_ms,
            experiment_ids=experiment_ids,
            transition_id=transition_id,
        )

    except Exception as e:
        # Wrap non-Sirius exceptions
        if not isinstance(e, PipelineError):
            raise PipelineError(
                f"Pipeline failed: {e}",
                context={"stage": reporter._stage.value, "transition_id": transition_id},
            ) from e
        raise


async def morph_async(
    image_a: str,
    image_b: str,
    **kwargs: Any,
) -> MorphResult:
    """Async version of morph.

    Runs the morph operation in an executor to avoid blocking.

    Args:
        image_a: Path to source image.
        image_b: Path to target image.
        **kwargs: Same arguments as morph().

    Returns:
        MorphResult with video path and data.
    """
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: morph(image_a, image_b, **kwargs))


# =============================================================================
# Streaming API (per Soumith Chintala feedback)
# =============================================================================


async def morph_stream(
    image_a: str,
    image_b: str,
    **kwargs: Any,
) -> tuple[MorphResult, list[Any]]:
    """Morph with streaming progress updates.

    Returns both the result and all progress updates.

    Args:
        image_a: Path to source image.
        image_b: Path to target image.
        **kwargs: Same arguments as morph().

    Returns:
        Tuple of (MorphResult, list of ProgressUpdates).

    Example:
        >>> result, updates = await morph_stream("a.png", "b.png")
        >>> for u in updates:
        ...     print(u)
    """
    from .progress import ProgressUpdate

    updates: list[ProgressUpdate] = []

    def collect_progress(update: ProgressUpdate) -> None:
        updates.append(update)

    # Merge callback with existing
    existing_callback = kwargs.pop("on_progress", None)

    def combined_callback(update: ProgressUpdate) -> None:
        collect_progress(update)
        if existing_callback:
            existing_callback(update)

    result = await morph_async(image_a, image_b, on_progress=combined_callback, **kwargs)
    return result, updates
