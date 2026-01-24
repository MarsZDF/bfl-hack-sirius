"""Editor module: Video assembly using imageio.

Per George Hotz feedback: Use imageio instead of FFmpeg subprocess.
One less system dependency, pure Python.
"""

from collections.abc import Sequence
from pathlib import Path

import imageio.v3 as iio
import numpy as np
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
    print(f"Assembling {len(sorted_frames)} frames in order: {[f.index for f in sorted_frames]}")

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


def create_fallback_video(
    image_a_path: str,
    image_b_path: str,
    output_path: str,
    frame_count: int = 16,
    config: VideoConfig | None = None,
) -> str:
    """Create a simple cross-dissolve video when generation fails.

    Args:
        image_a_path: Path to start image.
        image_b_path: Path to end image.
        output_path: Path for output video.
        frame_count: Number of frames to generate.
        config: Video configuration.

    Returns:
        Path to output video.
    """
    if config is None:
        config = VideoConfig()

    try:
        from PIL import ImageOps

        # Load images
        img_a = Image.open(image_a_path).convert("RGB")
        img_b = Image.open(image_b_path).convert("RGB")

        # Pad both images to the same size (use larger dimensions, no stretching)
        target_w = max(img_a.width, img_b.width)
        target_h = max(img_a.height, img_b.height)
        target_size = (target_w, target_h)

        img_a = ImageOps.pad(img_a, target_size, color="black", centering=(0.5, 0.5))
        img_b = ImageOps.pad(img_b, target_size, color="black", centering=(0.5, 0.5))

        # Generate frames
        frames = []
        for i in range(frame_count):
            alpha = i / max(1, frame_count - 1)
            # Cross-dissolve
            blended = Image.blend(img_a, img_b, alpha)
            frames.append(np.array(blended))

        # Handle boomerang
        if config.boomerang:
            frames += frames[-2:0:-1]

        # Ensure output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Write video
        iio.imwrite(
            output_path,
            frames,
            fps=config.framerate,
            codec=config.codec,
            quality=10 - (config.quality // 5),
        )
        return output_path

    except Exception as e:
        raise VideoEncodingError(f"Fallback video creation failed: {e}", output_path=output_path) from e


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


def add_audio(
    video_path: str,
    audio_path: str,
    output_path: str | None = None,
    fade_duration: float = 0.5,
) -> str:
    """Add audio track to video with fade in/out.

    Uses FFmpeg via imageio-ffmpeg (already installed).

    Args:
        video_path: Path to input video (mp4).
        audio_path: Path to audio file (mp3, wav, etc.) or URL.
        output_path: Output path. If None, replaces original with '_audio' suffix.
        fade_duration: Fade in/out duration in seconds.

    Returns:
        Path to output video with audio.

    Example:
        >>> from sirius.editor import add_audio
        >>> add_audio("morph.mp4", "ambient.mp3")
        'morph_audio.mp4'
    """
    import shutil
    import subprocess

    # Get ffmpeg path from imageio-ffmpeg
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        # Fallback to system ffmpeg
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise AssemblyError("FFmpeg not found. Install via: pip install imageio-ffmpeg")

    # Determine output path
    if output_path is None:
        video_p = Path(video_path)
        output_path = str(video_p.parent / f"{video_p.stem}_audio{video_p.suffix}")

    # Get video duration for audio fade
    probe_cmd = [
        ffmpeg_path, "-i", video_path,
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(
            probe_cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Parse duration from stderr (ffmpeg outputs info to stderr)
        import re
        duration_match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr)
        if duration_match:
            h, m, s = duration_match.groups()
            video_duration = int(h) * 3600 + int(m) * 60 + float(s)
        else:
            video_duration = 5.0  # Default fallback
    except Exception:
        video_duration = 5.0

    # Build FFmpeg command with audio fade
    fade_out_start = max(0, video_duration - fade_duration)
    audio_filter = f"afade=t=in:st=0:d={fade_duration},afade=t=out:st={fade_out_start}:d={fade_duration}"

    cmd = [
        ffmpeg_path,
        "-y",  # Overwrite output
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",  # Copy video stream (no re-encode)
        "-c:a", "aac",
        "-b:a", "128k",
        "-af", audio_filter,
        "-shortest",  # End when shortest stream ends
        "-movflags", "+faststart",  # Web-friendly
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return output_path
    except subprocess.CalledProcessError as e:
        raise AssemblyError(
            f"Audio muxing failed: {e.stderr.decode() if e.stderr else str(e)}",
            context={"video": video_path, "audio": audio_path},
        ) from e


# =============================================================================
# Free Ambient Audio URLs (royalty-free, no API key needed)
# =============================================================================

AMBIENT_AUDIO_URLS = {
    "calm": "https://cdn.pixabay.com/download/audio/2022/05/27/audio_1808fbf07a.mp3",  # Calm ambient
    "cinematic": "https://cdn.pixabay.com/download/audio/2022/03/15/audio_942237dca5.mp3",  # Cinematic rise
    "dreamy": "https://cdn.pixabay.com/download/audio/2023/07/05/audio_66269ef263.mp3",  # Dreamy pad
}


def add_ambient_audio(
    video_path: str,
    mood: str = "calm",
    output_path: str | None = None,
) -> str:
    """Add royalty-free ambient audio to video.

    Downloads audio from Pixabay (no API key required for direct links).

    Args:
        video_path: Path to input video.
        mood: One of "calm", "cinematic", "dreamy".
        output_path: Output path (optional).

    Returns:
        Path to video with audio.

    Example:
        >>> from sirius.editor import add_ambient_audio
        >>> add_ambient_audio("morph.mp4", mood="cinematic")
        'morph_audio.mp4'
    """
    import tempfile

    import requests

    if mood not in AMBIENT_AUDIO_URLS:
        raise ValueError(f"Unknown mood '{mood}'. Choose from: {list(AMBIENT_AUDIO_URLS.keys())}")

    audio_url = AMBIENT_AUDIO_URLS[mood]

    # Download audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        response = requests.get(audio_url, timeout=30)
        response.raise_for_status()
        tmp.write(response.content)
        tmp_audio = tmp.name

    try:
        return add_audio(video_path, tmp_audio, output_path)
    finally:
        # Clean up temp file
        Path(tmp_audio).unlink(missing_ok=True)
