"""Basic usage example of Sirius."""

import os
from sirius import morph

def main():
    # Ensure assets exist
    if not os.path.exists("assets/start.png") or not os.path.exists("assets/end.png"):
        print("Please place 'start.png' and 'end.png' in an 'assets' folder.")
        return

    print("Starting basic morph...")
    
    # Simple synchronous morph
    result = morph(
        "assets/start.png",
        "assets/end.png",
        output_dir="outputs/basic_example",
        frame_count=12,
        boomerang=True
    )
    
    print(f"Success! Video saved to: {result.video_path}")
    print(f"Total time: {result.duration_ms / 1000:.1f}s")

if __name__ == "__main__":
    main()
