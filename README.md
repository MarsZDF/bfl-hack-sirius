# Sirius 🌟

**AI Image Morphing Pipeline using Claude Vision + FLUX.2**

Sirius is a high-performance Python library that creates smooth, narrative-driven video transitions between two images. It uses **Claude Vision** to understand the content and style of your images, and **FLUX.2** (via BFL API) to generate high-quality intermediate frames that blend them seamlessly.

> Built for the BFL Hackathon (Jan 2026).

## ✨ Key Features

- **Intelligent Director**: Uses Claude Vision to analyze your start/end images and "screenwrite" a transition plan.
- **Narrative Transitions**: Supports multiple styles (`morph`, `narrative`, `fade`, `surreal`) to control *how* the change happens.
- **Parallel Generation**: Generates frames in parallel using the BFL FLUX.2 API (up to 4x faster).
- **High Quality**: Uses `flux-pro-1.1` for anchor frames and `flux-2-klein-9b` for fast, consistent animation.
- **Robust Pipeline**: Includes automatic retries, progress tracking (Codex-ready), and graceful error handling.

## 🚀 Installation

```bash
pip install -e .
```

### Prerequisites

You need API keys for:
1. **Anthropic** (Claude Vision): `ANTHROPIC_API_KEY`
2. **Black Forest Labs** (FLUX.2): `BFL_API_KEY`

Create a `.env` file in your project root:

```env
ANTHROPIC_API_KEY=sk-ant-...
BFL_API_KEY=bfl-...
```

## 📖 Usage

### Basic Morph

```python
from sirius import morph

# Create a smooth transition video
result = morph(
    "assets/cat.png",
    "assets/dog.png",
    output_dir="outputs"
)

print(f"Video created at: {result.video_path}")
```

### Advanced Control

```python
from sirius import morph, TransitionStyle

# Create a "surreal" dreamlike transition
result = morph(
    "start.jpg",
    "end.jpg",
    frame_count=24,
    transition_style=TransitionStyle.SURREAL,
    boomerang=True,  # Loop A -> B -> A
    seed=123
)
```

### Async & Streaming (for UI/Web)

```python
import asyncio
from sirius import morph_stream

async def main():
    image_a = "start.png"
    image_b = "end.png"

    # Stream progress updates
    result, updates = await morph_stream(image_a, image_b)
    
    for update in updates:
        print(f"{update.progress:.0%} - {update.message}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🏗 Architecture

Sirius is composed of three main agents (modules):

1.  **Director (`director.py`)**: 
    -   **Role**: The Creative Lead.
    -   **Tools**: Claude 3.5 Sonnet (Vision).
    -   **Task**: Sees both images, analyzes style/lighting/subject, and writes a detailed prompt sequence to bridge them.

2.  **Animator (`animator.py`)**: 
    -   **Role**: The Production Studio.
    -   **Tools**: BFL FLUX.2 API (Parallelized).
    -   **Task**: Takes the prompts and generates frames. Uses `flux-pro` for keyframes and `flux-klein` for in-betweens.

3.  **Editor (`editor.py`)**: 
    -   **Role**: Post-Production.
    -   **Tools**: `imageio` / `FFmpeg`.
    -   **Task**: Assembles frames into a high-quality MP4 or GIF.

## 🛠 Configuration

Configuration is handled via `_config.py` but can be overridden at runtime.

| Environment Variable | Description |
|----------------------|-------------|
| `ANTHROPIC_API_KEY` | Required for Director (Claude) |
| `BFL_API_KEY` | Required for Animator (FLUX) |

## 🧪 Testing

Run the test suite:

```bash
pytest
```