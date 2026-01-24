"""Animator module: Parallel FLUX.2 frame generation.

Per John Carmack feedback: Parallelize frame generation. With 16 frames
and 4 concurrent workers, you cut generation time by 4x.
"""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from ._bfl_client import BFLClient
from ._config import (
    ANIMATOR_MODEL,
    ANCHOR_MODEL,
    DEFAULT_GUIDANCE,
    DEFAULT_HEIGHT,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
    DEFAULT_WORKERS,
)
from ._types import Frame, GenerationConfig, TransitionPlan
from .exceptions import FrameGenerationError, GenerationError

# Type for progress callback during generation
FrameProgressCallback = Callable[[int, int, str], None]  # (current, total, status)


def generate_frame(
    index: int,
    prompt: str,
    config: GenerationConfig,
    output_dir: str,
    model: str,
    client: Any,
) -> Frame:
    """Generate a single frame."""
    try:
        # Check if client supports guidance/steps
        # Most models (except Flux Dev) don't use these in the same way,
        # but our clients should handle them or ignore them.
        image, duration_ms = client.generate(
            prompt=prompt,
            model=model,
            seed=config.seed,
            width=config.width,
            height=config.height,
            guidance=config.guidance,
            steps=config.steps,
        )

        # Save frame with padded index
        frame_path = Path(output_dir) / f"frame_{index:03d}.png"
        image.save(frame_path)

        return Frame(
            index=index,
            prompt=prompt,
            path=str(frame_path),
            duration_ms=duration_ms,
            model=model,
            seed=config.seed,
        )

    except Exception as e:
        raise FrameGenerationError(
            frame_index=index,
            prompt=prompt,
            reason=str(e),
            retryable=True,
        ) from e


def generate_frames_parallel(
    prompts: list[str],
    config: GenerationConfig,
    output_dir: str,
    model: str = ANIMATOR_MODEL,
    client: Any | None = None,
    on_progress: FrameProgressCallback | None = None,
) -> list[Frame]:
    """Generate frames in parallel using thread pool.

    Args:
        prompts: List of prompts for each frame.
        config: Generation configuration.
        output_dir: Directory to save frames.
        model: FLUX model to use.
        client: Optional BFL client (creates new if not provided).
        on_progress: Optional callback for progress updates.

    Returns:
        List of Frame objects in order.

    Raises:
        GenerationError: If generation fails.
    """
    if client is None:
        client = BFLClient()

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    frames: dict[int, Frame] = {}
    errors: list[FrameGenerationError] = []
    total = len(prompts)

    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        # Submit all generation tasks
        future_to_index = {
            executor.submit(
                generate_frame,
                index=i,
                prompt=prompt,
                config=config,
                output_dir=output_dir,
                model=model,
                client=client,
            ): i
            for i, prompt in enumerate(prompts)
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                frame = future.result()
                frames[index] = frame
                completed += 1

                if on_progress:
                    on_progress(completed, total, f"Generated frame {index + 1}/{total}")

            except FrameGenerationError as e:
                errors.append(e)
                completed += 1

                if on_progress:
                    on_progress(completed, total, f"Frame {index + 1} failed: {e}")

    # Check for errors
    if errors:
        # If too many failures, raise
        if len(errors) > len(prompts) // 4:
            raise GenerationError(
                f"Too many frame failures: {len(errors)}/{len(prompts)}",
                context={"errors": [str(e) for e in errors[:5]]},
            )
        # Otherwise log but continue with what we have
        for e in errors:
            print(f"Warning: {e}")

    # Return frames in order
    return [frames[i] for i in sorted(frames.keys())]


def generate_anchors(
    prompt_a: str,
    prompt_b: str,
    config: GenerationConfig,
    output_dir: str,
    client: Any | None = None,
) -> tuple[Frame, Frame]:
    """Generate anchor frames (first and last) with high-quality model.

    Uses flux-pro-1.1 for better quality on the bookend frames.

    Args:
        prompt_a: Prompt for first frame.
        prompt_b: Prompt for last frame.
        config: Generation configuration.
        output_dir: Directory to save frames.
        client: Optional BFL client.

    Returns:
        Tuple of (first_frame, last_frame).
    """
    if client is None:
        client = BFLClient()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate both anchors (could parallelize but usually just 2)
    frame_a = generate_frame(
        index=0,
        prompt=prompt_a,
        config=config,
        output_dir=output_dir,
        model=ANCHOR_MODEL,
        client=client,
    )

    frame_b = generate_frame(
        index=-1,  # Will be renumbered later
        prompt=prompt_b,
        config=config,
        output_dir=output_dir,
        model=ANCHOR_MODEL,
        client=client,
    )

    return frame_a, frame_b


def animate(
    plan: TransitionPlan,
    config: GenerationConfig | None = None,
    output_dir: str = "outputs/frames",
    use_anchors: bool = True,
    animator_model: str = ANIMATOR_MODEL,
    anchor_model: str = ANCHOR_MODEL,
    client: Any | None = None,
    on_progress: FrameProgressCallback | None = None,
) -> list[Frame]:
    """Full animation flow: generate all frames from transition plan.

    This is the main entry point for Step 2 of the pipeline.

    Args:
        plan: TransitionPlan from Director.
        config: Generation configuration.
        output_dir: Directory to save frames.
        use_anchors: Whether to use high-quality model for first/last frames.
        client: Optional BFL client.
        on_progress: Optional progress callback.

    Returns:
        List of all generated frames in order.
    """
    if config is None:
        config = GenerationConfig()

    if client is None:
        client = BFLClient()

    # Create unique output directory for this animation
    transition_id = uuid.uuid4().hex[:8]
    frames_dir = Path(output_dir) / transition_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    all_frames: list[Frame] = []

    if use_anchors and len(plan.prompts) >= 2:
        # Generate anchor frames with high-quality model
        if on_progress:
            on_progress(0, plan.frame_count, "Generating anchor frames...")

        anchor_a = generate_frame(
            index=0,
            prompt=plan.prompts[0],
            config=config,
            output_dir=str(frames_dir),
            model=anchor_model,
            client=client,
        )
        all_frames.append(anchor_a)

        if on_progress:
            on_progress(1, plan.frame_count, "Generated first anchor")

        # Generate middle frames in parallel
        middle_prompts = plan.prompts[1:-1]
        if middle_prompts:
            middle_frames = generate_frames_parallel(
                prompts=middle_prompts,
                config=config,
                output_dir=str(frames_dir),
                model=animator_model,
                client=client,
                on_progress=lambda c, t, s: on_progress(
                    1 + c, plan.frame_count, s
                ) if on_progress else None,
            )

            # Renumber middle frames
            for i, frame in enumerate(middle_frames):
                frame.index = i + 1
                # Rename file to match index
                old_path = Path(frame.path)
                new_path = frames_dir / f"frame_{frame.index:03d}.png"
                if old_path != new_path:
                    old_path.rename(new_path)
                    frame.path = str(new_path)

            all_frames.extend(middle_frames)

        # Generate last anchor
        anchor_b = generate_frame(
            index=plan.frame_count - 1,
            prompt=plan.prompts[-1],
            config=config,
            output_dir=str(frames_dir),
            model=anchor_model,
            client=client,
        )
        all_frames.append(anchor_b)

        if on_progress:
            on_progress(plan.frame_count, plan.frame_count, "All frames generated")

    else:
        # Generate all frames with same model
        all_frames = generate_frames_parallel(
            prompts=plan.prompts,
            config=config,
            output_dir=str(frames_dir),
            model=animator_model,
            client=client,
            on_progress=on_progress,
        )

    # Sort by index to ensure order
    all_frames.sort(key=lambda f: f.index)

    return all_frames


def create_config(
    seed: int = DEFAULT_SEED,
    guidance: float = DEFAULT_GUIDANCE,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    steps: int = DEFAULT_STEPS,
    workers: int = DEFAULT_WORKERS,
) -> GenerationConfig:
    """Create generation configuration with defaults.

    Args:
        seed: Random seed for reproducibility.
        guidance: Guidance scale.
        width: Output width.
        height: Output height.
        steps: Inference steps.
        workers: Number of parallel workers.

    Returns:
        GenerationConfig instance.
    """
    return GenerationConfig(
        seed=seed,
        guidance=guidance,
        width=width,
        height=height,
        steps=steps,
        workers=workers,
    )
