"""Pipeline orchestration: ties Director, Animator, and Editor together.

Main entry point for the Sirius morph operation.
"""

import time
import uuid
from pathlib import Path
from typing import Any

from ._anthropic_client import ClaudeClient
from ._bfl_client import BFLClient
from ._runware_client import RunwareClient
from ._config import (
    DEFAULT_FRAME_COUNT,
    DEFAULT_SEED,
    RUNWARE_FLUX_PRO,
    RUNWARE_FLUX_DEV,
    RUNWARE_FLUX_SCHNELL,
    get_runware_api_key,
)
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
from .editor import edit, create_preview, create_fallback_video
from .exceptions import PipelineError, GenerationError
from .progress import (
    CancellationToken,
    MorphStage,
    ProgressCallback,
    ProgressReporter,
    create_reporter,
)


def plan_morph(
    image_a: str,
    image_b: str,
    frame_count: int = DEFAULT_FRAME_COUNT,
    transition_style: TransitionStyle | str = TransitionStyle.MORPH,
    user_context: str | None = None,
) -> TransitionPlan:
    """Generate a transition plan without executing it.

    Useful for "Director's Review" workflow where you want to approve/edit
    prompts before generating frames.

    Args:
        image_a: Path to source image.
        image_b: Path to target image.
        frame_count: Number of frames to generate.
        transition_style: Style of transition.

    Returns:
        TransitionPlan object.
    """
    return direct(
        image_a_path=image_a,
        image_b_path=image_b,
        frame_count=frame_count,
        style=transition_style,
        user_context=user_context,
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
    preview: bool = False,
    on_progress: ProgressCallback | None = None,
    cancellation: CancellationToken | None = None,
    config: GenerationConfig | None = None,
    plan: TransitionPlan | None = None,
    user_context: str | None = None,
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

    # Choose image generation client
    runware_key = get_runware_api_key()
    gen_client: Any
    if runware_key:
        gen_client = RunwareClient(runware_key)
        # Use Runware model IDs
        if preview:
            anim_model = RUNWARE_FLUX_SCHNELL
            anch_model = RUNWARE_FLUX_SCHNELL # No anchors in preview, but just in case
        else:
            anim_model = RUNWARE_FLUX_DEV
            anch_model = RUNWARE_FLUX_PRO
    else:
        gen_client = BFLClient()
        from ._config import ANCHOR_MODEL, ANIMATOR_MODEL
        # Fallback for BFL (maybe use Klein for preview if available, but Schnell is best)
        # For now, just use Klein for preview on BFL side if available or stick to default
        if preview:
             from ._config import FLUX_KLEIN
             anim_model = FLUX_KLEIN
             anch_model = FLUX_KLEIN
        else:
             anim_model = ANIMATOR_MODEL
             anch_model = ANCHOR_MODEL

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
        if plan is None:
            reporter.start_stage(MorphStage.ANALYZING)

            plan = direct(
                image_a_path=image_a,
                image_b_path=image_b,
                frame_count=frame_count,
                style=transition_style,
                client=claude_client,
                user_context=user_context,
            )

            reporter.start_stage(MorphStage.PLANNING, "Transition plan created")
        else:
            reporter.start_stage(MorphStage.PLANNING, "Using existing transition plan")

        # =================================================================
        # STEP 2: ANIMATOR - Generate frames in parallel
        # =================================================================
        reporter.start_stage(MorphStage.GENERATING)

        def animator_progress(current: int, total: int, message: str) -> None:
            reporter.update_frame(current, total, message)

        try:
            frames = animate(
                plan=plan,
                config=config,
                output_dir=str(frames_dir),
                use_anchors=not preview,
                preview=preview,
                animator_model=anim_model,
                anchor_model=anch_model,
                client=gen_client,
                on_progress=animator_progress,
                image_a=image_a if not preview else None,
                image_b=image_b if not preview else None,
            )
        except GenerationError as e:
            # Fallback: Create cross-dissolve video
            reporter.update(MorphStage.GENERATING, f"Generation failed. Switching to fallback... ({e})")
            
            video_filename = video_name or f"fallback_{transition_id}.mp4"
            video_path = str(videos_dir / video_filename)
            video_config = VideoConfig(boomerang=boomerang)
            
            create_fallback_video(
                image_a, 
                image_b, 
                video_path, 
                frame_count, 
                video_config
            )
            
            reporter.complete(f"Fallback video created: {video_path}")
            
            return MorphResult(
                video_path=video_path,
                frames=[],
                plan=plan,
                duration_ms=int((time.time() - start_time) * 1000),
                transition_id=transition_id,
            )

        # =================================================================
        # STEP 3: EDITOR - Assemble video OR Preview
        # =================================================================
        reporter.start_stage(MorphStage.ASSEMBLING)

        if preview:
            preview_filename = video_name or f"preview_{transition_id}.jpg"
            # Force extension
            if not preview_filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                 preview_filename += ".jpg"
            
            video_path = str(base_output / preview_filename)
            create_preview(frames, video_path)
            reporter.complete(f"Preview created: {video_path}")
        else:
            video_filename = video_name or f"morph_{transition_id}.mp4"
            video_path = str(videos_dir / video_filename)

            video_config = VideoConfig(boomerang=boomerang)
            edit(frames, video_path, video_config)
            reporter.complete(f"Video created: {video_path}")

        # =================================================================
        # COMPLETE
        # =================================================================
        duration_ms = int((time.time() - start_time) * 1000)
        # Reporter already completed in step 3

        return MorphResult(
            video_path=video_path,
            frames=frames,
            plan=plan,
            duration_ms=duration_ms,
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
