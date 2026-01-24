"""Example showing all transition styles."""

import os
from sirius import morph, TransitionStyle

def main():
    if not os.path.exists("assets/start.png"):
        print("Please provide assets/start.png and assets/end.png")
        return

    styles = [
        TransitionStyle.MORPH,
        TransitionStyle.NARRATIVE,
        TransitionStyle.FADE,
        TransitionStyle.SURREAL
    ]

    for style in styles:
        print(f"\n--- Running style: {style.value} ---")
        
        result = morph(
            "assets/start.png",
            "assets/end.png",
            transition_style=style,
            frame_count=8,  # Fewer frames for quick demo
            output_dir="outputs/styles_example",
            video_name=f"morph_{style.value}.mp4"
        )
        
        print(f"Created: {result.video_path}")

if __name__ == "__main__":
    main()

