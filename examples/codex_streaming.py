"""Example of streaming progress updates (Codex pattern)."""

import asyncio
import os
from sirius import morph_stream

async def main():
    if not os.path.exists("assets/start.png"):
        print("Please provide assets/start.png and assets/end.png")
        return

    print("Starting async morph with streaming updates...")

    # Using async iterator pattern
    result, updates = await morph_stream(
        "assets/start.png",
        "assets/end.png",
        frame_count=12,
        output_dir="outputs/streaming_example"
    )
    
    # In a real app, you'd iterate async:
    # async for update in progress_stream(reporter): ...
    # But morph_stream collects them for you if you just want the result + logs.
    
    # We can also pass a callback to morph_stream for real-time printing
    # But here let's inspect the collected updates
    
    for update in updates:
        print(f"[{update.stage.value}] {update.progress:.1%} - {update.message}")
        if update.eta_ms:
            print(f"  ETA: {update.eta_ms/1000:.1f}s")

    print(f"\nFinished: {result.video_path}")

if __name__ == "__main__":
    asyncio.run(main())

