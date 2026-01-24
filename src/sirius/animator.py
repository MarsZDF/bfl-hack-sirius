"""Animator module: Parallel FLUX.2 frame generation.

Per John Carmack feedback: Parallelize frame generation. With 16 frames
and 4 concurrent workers, you cut generation time by 4x.
"""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageOps

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
                    on_progress(completed, total, f"Generated frame {index + 1} ({completed}/{total})")

            except FrameGenerationError as e:
                errors.append(e)
                completed += 1

                if on_progress:
                    on_progress(completed, total, f"Frame {index + 1} failed: {e}")

    # Check for errors
    if errors:
        success_count = len(frames)
        failure_count = len(errors)
        
        # If complete failure, raise error
        if success_count == 0:
            raise GenerationError(
                f"All {failure_count} frames failed to generate",
                context={"errors": [str(e) for e in errors[:5]]},
            )
            
        # If partial failure, warn but proceed
        print(f"⚠️ Warning: {failure_count}/{total} frames failed. Video may be jumpy.")
        for e in errors:
            print(f"  - {e}")

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
    preview: bool = False,
    animator_model: str = ANIMATOR_MODEL,
    anchor_model: str = ANCHOR_MODEL,
    client: Any | None = None,
    on_progress: FrameProgressCallback | None = None,
    image_a: str | None = None,
    image_b: str | None = None,
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

    # Use output directory directly (pipeline manages uniqueness)
    frames_dir = Path(output_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    all_frames: list[Frame] = []
    
    # 1. Handle First Frame (Anchor A)
    if image_a:
        # Use original image
        if on_progress:
            on_progress(0, plan.frame_count, "Processing start image...")
        
        img = Image.open(image_a).convert("RGB")
        # Resize preserving aspect ratio and pad
        img.thumbnail((config.width, config.height), Image.Resampling.LANCZOS)
        img = ImageOps.pad(img, (config.width, config.height), color="black")
        
        path_a = frames_dir / "frame_000.png"
        img.save(path_a)
        
        all_frames.append(Frame(
            index=0,
            prompt=plan.prompts[0],
            path=str(path_a),
            duration_ms=0,
            model="original",
            seed=0
        ))
    elif use_anchors and not preview:
        # Generate with Anchor Model
        if on_progress:
            on_progress(0, plan.frame_count, "Generating start anchor...")
        
        all_frames.append(generate_frame(
            index=0,
            prompt=plan.prompts[0],
            config=config,
            output_dir=str(frames_dir),
            model=anchor_model,
            client=client,
        ))
    
    # 2. Handle Middle Frames (Parallel)
    # If we have start/end images, we generate frames 1 to N-2.
    # If we don't, we might generate 0 to N-1 or 1 to N-2 depending on anchor logic.
    # To keep it simple: parallelize everything that ISN'T an anchor we just handled.
    
    # Indices to generate in parallel
    start_idx = 1
    end_idx = plan.frame_count - 2
    
    if not image_a and not (use_anchors and not preview):
        start_idx = 0 # Generate frame 0 in parallel if no special handling
        
    if not image_b and not (use_anchors and not preview):
        end_idx = plan.frame_count - 1 # Generate last frame in parallel
        
    middle_prompts = []
    middle_indices = []
    
    for i in range(start_idx, end_idx + 1):
        if i < len(plan.prompts):
            middle_prompts.append(plan.prompts[i])
            middle_indices.append(i)
            
    if middle_prompts:
        # Map generated results back to correct indices
        # generate_frames_parallel assumes 0..N indices in return list
        # We need to manually call generate_frame or adjust generate_frames_parallel
        # Actually generate_frames_parallel takes prompts and returns a list.
        # We can't easily map indices unless we modify it or call generate_frame directly.
        
        # Let's use ThreadPoolExecutor directly here for flexibility
        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            future_to_index = {
                executor.submit(
                    generate_frame,
                    index=idx,
                    prompt=p,
                    config=config,
                    output_dir=str(frames_dir),
                    model=animator_model,
                    client=client,
                ): idx
                for idx, p in zip(middle_indices, middle_prompts)
            }
            
            completed = 0
            total_middle = len(middle_indices)
            
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    frame = future.result()
                    all_frames.append(frame)
                    completed += 1
                    if on_progress:
                        on_progress(completed, total_middle, f"Generated frame {idx + 1} ({completed}/{total_middle})")
                except FrameGenerationError as e:
                    print(f"Warning: Frame {idx} failed: {e}")
                    # Soft failure: don't append

    # 3. Handle Last Frame (Anchor B)
    if image_b:
        # Use original image
        img = Image.open(image_b).convert("RGB")
        # Resize preserving aspect ratio and pad
        img.thumbnail((config.width, config.height), Image.Resampling.LANCZOS)
        img = ImageOps.pad(img, (config.width, config.height), color="black")
        
        path_b = frames_dir / f"frame_{plan.frame_count - 1:03d}.png"
        img.save(path_b)
        
        all_frames.append(Frame(
            index=plan.frame_count - 1,
            prompt=plan.prompts[-1] if plan.prompts else "",
            path=str(path_b),
            duration_ms=0,
            model="original",
            seed=0
        ))
    elif use_anchors and not preview:
        # Generate with Anchor Model
        if on_progress:
            on_progress(plan.frame_count, plan.frame_count, "Generating end anchor...")
            
        all_frames.append(generate_frame(
            index=plan.frame_count - 1,
            prompt=plan.prompts[-1],
            config=config,
            output_dir=str(frames_dir),
            model=anchor_model,
            client=client,
        ))

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
