"""Example of generating a preview grid instead of just video."""

import os
from sirius import morph, create_preview

def main():
    if not os.path.exists("assets/start.png"):
        print("Please provide assets/start.png and assets/end.png")
        return

    print("Generating frames...")
    
    # Run morph to get frames
    result = morph(
        "assets/start.png",
        "assets/end.png",
        frame_count=16,
        output_dir="outputs/preview_example"
    )
    
    print("Creating preview grid...")
    
    # Create static image grid
    preview_path = "outputs/preview_example/grid_preview.jpg"
    create_preview(result.frames, preview_path, cols=4)
    
    print(f"Preview saved to: {preview_path}")

if __name__ == "__main__":
    main()
