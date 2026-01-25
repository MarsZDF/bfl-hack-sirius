"""CLI entry point for Sirius.

Usage:
    python -m sirius image_a.png image_b.png
    python -m sirius image_a.png image_b.png -o output.mp4
    python -m sirius image_a.png image_b.png --style surreal --frames 24
"""

import argparse
import sys
import time

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from . import AspectRatio, GenerationConfig, TransformType, TransitionStyle, __version__, morph
from .director import describe_changes, transform
from .editor import add_ambient_audio, add_audio
from .progress import MorphStage, ProgressUpdate

console = Console()


def create_progress_callback(progress: Progress, task_id: int, stats: dict) -> callable:
    """Create a progress callback that updates Rich progress bar."""
    def callback(update: ProgressUpdate) -> None:
        progress.update(task_id, completed=update.progress * 100, description=update.message)

        # Track frame timing for speedup calculation
        if update.stage == MorphStage.GENERATING and update.current_frame:
            if "first_frame_time" not in stats:
                stats["first_frame_time"] = time.time()
            stats["last_frame_time"] = time.time()
            stats["frames_generated"] = update.current_frame
            stats["total_frames"] = update.total_frames

    return callback


def print_speedup_stats(stats: dict, total_time: float) -> None:
    """Print speedup statistics."""
    if "frames_generated" not in stats:
        return

    frames = stats.get("frames_generated", 0)
    total_frames = stats.get("total_frames", frames)

    if frames == 0:
        return

    # Calculate actual generation time (excluding analysis/assembly)
    gen_time = stats.get("last_frame_time", 0) - stats.get("first_frame_time", 0)
    if gen_time <= 0:
        gen_time = total_time * 0.75  # Estimate ~75% of time is generation

    # Estimate sequential time (~3s per frame is typical for FLUX)
    estimated_sequential = frames * 3.0

    # Calculate speedup
    if gen_time > 0:
        fps = frames / gen_time
        speedup = estimated_sequential / gen_time

        console.print()
        console.print("[bold green]⚡ Performance Stats[/bold green]")
        console.print(f"   Frames: {frames}/{total_frames}")
        console.print(f"   Generation time: {gen_time:.1f}s ({fps:.2f} fps)")
        console.print(f"   Sequential estimate: ~{estimated_sequential:.0f}s")
        console.print(f"   [bold cyan]Speedup: {speedup:.1f}x[/bold cyan] (parallel generation)")


def handle_transform(args) -> int:
    """Handle single-image transform mode."""
    from pathlib import Path
    from .animator import animate, create_config

    transform_type = TransformType(args.transform)
    aspect = AspectRatio(args.aspect)
    width, height = aspect.get_dimensions()

    console.print(f"[bold blue]🌟 Sirius v{__version__} - Transform Mode[/bold blue]")
    console.print(f"   Image:     {args.image_a}")
    console.print(f"   Transform: {transform_type.value}")
    console.print(f"   Aspect:    {args.aspect} ({width}x{height})")
    console.print(f"   Frames:    {args.frames}")
    console.print()

    # Determine output
    output_dir = "outputs"
    video_name = None
    if args.output:
        output_path = Path(args.output)
        output_dir = str(output_path.parent) if output_path.parent != Path(".") else "outputs"
        video_name = output_path.name

    stats: dict = {}
    start_time = time.time()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Planning transformation...", total=100)

            # Get transform plan
            plan = transform(
                args.image_a,
                transform_type=transform_type,
                frame_count=args.frames,
            )
            progress.update(task, completed=20, description="Generating frames...")

            # Create generation config
            gen_config = GenerationConfig(
                width=width,
                height=height,
                seed=args.seed,
            )

            # Generate frames
            def on_frame_progress(current: int, total: int, msg: str) -> None:
                pct = 20 + (current / total) * 70
                progress.update(task, completed=pct, description=msg)

            frames = animate(
                plan=plan,
                config=gen_config,
                output_dir=str(Path(output_dir) / "frames"),
                use_anchors=not args.preview,
                preview=args.preview,
                on_progress=on_frame_progress,
                image_a=args.image_a,
            )

            progress.update(task, completed=90, description="Assembling video...")

            # Assemble video
            from .editor import edit, create_preview as make_preview
            from ._types import VideoConfig

            if args.preview:
                video_filename = video_name or f"transform_preview.jpg"
                video_path = str(Path(output_dir) / video_filename)
                make_preview(frames, video_path)
            else:
                video_filename = video_name or f"transform_{transform_type.value}.mp4"
                video_path = str(Path(output_dir) / "videos" / video_filename)
                Path(video_path).parent.mkdir(parents=True, exist_ok=True)
                video_config = VideoConfig(boomerang=args.boomerang)
                edit(frames, video_path, video_config)

            progress.update(task, completed=100, description="Done!")

        total_time = time.time() - start_time

        console.print()
        console.print(f"[bold green]✨ Done![/bold green] Total time: {total_time:.1f}s")
        console.print(f"   Output: [link=file://{video_path}]{video_path}[/link]")

        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="sirius",
        description="AI Image Morphing Pipeline - Transform images with Claude Vision + FLUX.2",
        epilog="Example: python -m sirius cat.png dog.png -o morph.mp4",
    )

    parser.add_argument("image_a", help="Path to source image")
    parser.add_argument("image_b", nargs="?", help="Path to target image (optional if using --transform)")
    parser.add_argument("-o", "--output", help="Output video path (default: outputs/morph.mp4)")
    parser.add_argument(
        "-s", "--style",
        choices=["morph", "narrative", "fade", "surreal", "timelapse", "metamorphosis", "glitch", "painterly"],
        default="morph",
        help="Transition style (default: morph)",
    )
    parser.add_argument(
        "-t", "--transform",
        choices=["age", "seasons", "bloom", "sunrise", "dream"],
        help=(
            "Single-image transform (no end image needed). "
            "age: show subject growing up (always positive!), "
            "seasons: cycle through seasons, "
            "bloom: growth and flourishing, "
            "sunrise: night to dawn, "
            "dream: transform into magical version"
        ),
    )
    parser.add_argument(
        "-a", "--aspect",
        choices=["16:9", "9:16", "1:1", "21:9", "4:3"],
        default="9:16",
        help=(
            "Aspect ratio (default: 9:16 for social). Options: "
            "9:16 (TikTok/Reels/Stories), "
            "16:9 (YouTube/desktop), "
            "1:1 (Instagram feed), "
            "21:9 (cinematic), "
            "4:3 (classic photo)"
        ),
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Describe what changed between images (no video generation)",
    )
    parser.add_argument(
        "-f", "--frames",
        type=int,
        default=16,
        help="Number of frames to generate (default: 16)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Fast preview mode (lower quality, no video)",
    )
    parser.add_argument(
        "--boomerang",
        action="store_true",
        help="Loop video A→B→A",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--context",
        type=str,
        help="Additional instructions for the AI director",
    )
    parser.add_argument(
        "--audio",
        type=str,
        help="Path to audio file to add to video",
    )
    parser.add_argument(
        "--mood",
        choices=["calm", "cinematic", "dreamy"],
        help="Add royalty-free ambient audio (alternative to --audio)",
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"sirius {__version__}",
    )

    args = parser.parse_args()

    # Validate: if no transform mode, image_b is required
    if not args.transform and not args.image_b and not args.describe:
        parser.error("image_b is required (or use --transform for single-image mode)")

    # Handle --describe mode (no video generation)
    if args.describe:
        if not args.image_b:
            parser.error("--describe requires both image_a and image_b")
        console.print("[bold blue]🔍 Analyzing changes...[/bold blue]")
        console.print(f"   Before: {args.image_a}")
        console.print(f"   After:  {args.image_b}")
        console.print()

        try:
            result = describe_changes(args.image_a, args.image_b)

            console.print(f"[bold green]✨ {result['headline']}[/bold green]")
            console.print()
            console.print(result["description"])
            console.print()
            console.print(f"[dim]Time passed:[/dim] {result['time_passed']}")
            console.print(f"[dim]Mood shift:[/dim] {result['mood_shift']}")
            console.print()
            console.print("[bold]Changes:[/bold]")
            for change in result.get("changes", []):
                console.print(f"  • {change}")

            return 0
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            return 1

    # Handle --transform mode (single image)
    if args.transform:
        return handle_transform(args)

    # Parse style and aspect ratio
    style = TransitionStyle(args.style)
    aspect = AspectRatio(args.aspect)
    width, height = aspect.get_dimensions()

    # Determine output
    output_dir = "outputs"
    video_name = None
    if args.output:
        from pathlib import Path
        output_path = Path(args.output)
        output_dir = str(output_path.parent) if output_path.parent != Path(".") else "outputs"
        video_name = output_path.name

    console.print(f"[bold blue]🌟 Sirius v{__version__}[/bold blue]")
    console.print(f"   Source: {args.image_a}")
    console.print(f"   Target: {args.image_b}")
    console.print(f"   Style:  {style.value}")
    console.print(f"   Aspect: {args.aspect} ({width}x{height})")
    console.print(f"   Frames: {args.frames}")
    console.print()

    # Create config with aspect ratio dimensions
    gen_config = GenerationConfig(
        width=width,
        height=height,
        seed=args.seed,
    )

    stats: dict = {}
    start_time = time.time()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Initializing...", total=100)
            callback = create_progress_callback(progress, task, stats)

            result = morph(
                args.image_a,
                args.image_b,
                frame_count=args.frames,
                transition_style=style,
                preview=args.preview,
                boomerang=args.boomerang,
                seed=args.seed,
                user_context=args.context,
                output_dir=output_dir,
                video_name=video_name,
                on_progress=callback,
                config=gen_config,
            )

        total_time = time.time() - start_time

        # Print speedup stats
        print_speedup_stats(stats, total_time)

        # Add audio if requested
        final_path = result.video_path
        if args.audio or args.mood:
            console.print()
            console.print("[bold blue]🎵 Adding audio...[/bold blue]")
            try:
                if args.audio:
                    final_path = add_audio(result.video_path, args.audio)
                elif args.mood:
                    final_path = add_ambient_audio(result.video_path, mood=args.mood)
                console.print(f"   Audio added: {args.audio or args.mood}")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not add audio: {e}[/yellow]")
                final_path = result.video_path

        console.print()
        console.print(f"[bold green]✨ Done![/bold green] Total time: {total_time:.1f}s")
        console.print(f"   Output: [link=file://{final_path}]{final_path}[/link]")

        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
