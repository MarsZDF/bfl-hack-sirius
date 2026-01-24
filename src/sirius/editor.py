"""Editor module: Video assembly using imageio.

Per George Hotz feedback: Use imageio instead of FFmpeg subprocess.
One less system dependency, pure Python.
"""

from pathlib import Path
from typing import Sequence

import imageio.v3 as iio
from PIL import Image

from ._types import Frame, VideoConfig
from .exceptions import AssemblyError, FrameOrderError, VideoEncodingError


def validate_frames(frames: Sequence[Frame]) -> None:
    """Validate frames are in correct order and all exist.

    Args:
        frames: Sequence of Frame objects.

    Raises:
        FrameOrderError: If frames are missing or out of order.
        AssemblyError: If frame files don't exist.
    """
    if not frames:
        raise FrameOrderError(expected=1, actual=0)

    # Check all files exist
    for frame in frames:
        if not Path(frame.path).exists():
            raise AssemblyError(
                f"Frame file not found: {frame.path}",
                context={"frame_index": frame.index},
            )

    # Check indices are sequential (allowing gaps from failed generations)
    indices = sorted(f.index for f in frames)
    expected = list(range(min(indices), max(indices) + 1))
    missing = set(expected) - set(indices)

    if missing and len(missing) > len(frames) // 4:
        raise FrameOrderError(
            expected=len(expected),
            actual=len(frames),
            missing=sorted(missing),
        )


def load_frames_as_array(frames: Sequence[Frame]) -> list[Image.Image]:
    """Load frame images in order.

    Args:
        frames: Sequence of Frame objects (will be sorted by index).

    Returns:
        List of PIL Images in frame order.
    """
    sorted_frames = sorted(frames, key=lambda f: f.index)
    images = []

    for frame in sorted_frames:
        img = Image.open(frame.path)
        # Ensure RGB mode for video encoding
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(img)

    return images


def assemble_video(
    frames: Sequence[Frame],
    output_path: str,
    config: VideoConfig | None = None,
) -> str:
    """Combine frames into video using imageio.

    Args:
        frames: Sequence of Frame objects in order.
        output_path: Output video file path.
        config: Video configuration.

    Returns:
        Path to created video file.

    Raises:
        AssemblyError: If video assembly fails.
    """
    if config is None:
        config = VideoConfig()

    # Validate frames
    validate_frames(frames)

    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort frames by index
    sorted_frames = sorted(frames, key=lambda f: f.index)

    # Handle boomerang mode (A→B→A)
    if config.boomerang:
        # Add reversed frames (excluding first and last to avoid duplicates)
        reversed_frames = list(reversed(sorted_frames[1:-1]))
        sorted_frames = sorted_frames + reversed_frames

    try:
        # Collect frame paths
        frame_paths = [f.path for f in sorted_frames]

        # Read frames as numpy arrays
        frame_arrays = []
        for path in frame_paths:
            # Read and ensure consistent format
            img = iio.imread(path)
            # imageio returns numpy array, ensure it's uint8
            frame_arrays.append(img)

        # Write video
        iio.imwrite(
            output_path,
            frame_arrays,
            fps=config.framerate,
            codec=config.codec,
            quality=10 - (config.quality // 5),  # Convert CRF to quality (rough mapping)
        )

        return output_path

    except Exception as e:
        raise VideoEncodingError(str(e), output_path=output_path) from e


def assemble_gif(
    frames: Sequence[Frame],
    output_path: str,
    fps: int = 12,
    loop: int = 0,
    boomerang: bool = False,
) -> str:
    """Assemble frames into an animated GIF.

    Args:
        frames: Sequence of Frame objects.
        output_path: Output GIF path.
        fps: Frames per second.
        loop: Number of loops (0 = infinite).
        boomerang: Whether to create A→B→A loop.

    Returns:
        Path to created GIF.
    """
    validate_frames(frames)

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_frames = sorted(frames, key=lambda f: f.index)

    if boomerang:
        reversed_frames = list(reversed(sorted_frames[1:-1]))
        sorted_frames = sorted_frames + reversed_frames

    try:
        frame_paths = [f.path for f in sorted_frames]
        frame_arrays = [iio.imread(path) for path in frame_paths]

        # Calculate duration in milliseconds
        duration = int(1000 / fps)

        iio.imwrite(
            output_path,
            frame_arrays,
            duration=duration,
            loop=loop,
        )

        return output_path

    except Exception as e:
        raise VideoEncodingError(f"GIF creation failed: {e}", output_path=output_path) from e


def create_preview(
    frames: Sequence[Frame],
    output_path: str,
    thumbnail_size: int = 256,
    cols: int = 4,
) -> str:
    """Create a preview grid of all frames.

    Args:
        frames: Sequence of Frame objects.
        output_path: Output image path.
        thumbnail_size: Size of each thumbnail.
        cols: Number of columns in grid.

    Returns:
        Path to preview image.
    """
    validate_frames(frames)
    sorted_frames = sorted(frames, key=lambda f: f.index)

    # Calculate grid dimensions
    n_frames = len(sorted_frames)
    rows = (n_frames + cols - 1) // cols

    # Create canvas
    canvas_width = cols * thumbnail_size
    canvas_height = rows * thumbnail_size
    canvas = Image.new("RGB", (canvas_width, canvas_height), (30, 30, 30))

    # Place thumbnails
    for i, frame in enumerate(sorted_frames):
        img = Image.open(frame.path)
        img.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)

        row = i // cols
        col = i % cols
        x = col * thumbnail_size + (thumbnail_size - img.width) // 2
        y = row * thumbnail_size + (thumbnail_size - img.height) // 2

        canvas.paste(img, (x, y))

    canvas.save(output_path)
    return output_path


def edit(
    frames: Sequence[Frame],
    output_path: str,
    config: VideoConfig | None = None,
) -> str:
    """Main entry point for Step 3: assemble video from frames.

    Args:
        frames: Generated frames from Animator.
        output_path: Output video file path.
        config: Video configuration.

    Returns:
        Path to output video.
    """
    return assemble_video(frames, output_path, config)
