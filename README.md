# Sirius 🌟

**The AI Cinematographer: Seamless Image Morphing with Claude + FLUX.2**

> 🏆 **Built for the BFL Hackathon (Jan 2026)**

Sirius is a high-performance Python library that transforms any two static images into a smooth, narrative-driven video transition. It combines **Claude 3.5 Sonnet (Vision)** to "direct" the scene and **FLUX.2** (via Runware/BFL) to "animate" high-fidelity frames.

<div align="center">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  <br/>
  <b>TikTok-Ready (9:16) • Parallel Generation • Interactive Director</b>
</div>

## ✨ Key Features

-   📱 **TikTok-First**: Defaults to **9:16 (720x1280)** vertical video, perfect for Reels/Shorts.
-   🎬 **AI Director**: Uses **Claude Vision** to analyze your images and write a frame-by-frame screenplay, ensuring narrative consistency.
-   ⚡ **Parallel Rendering**: Generates 16 frames in parallel using an async **Runware** client, cutting render time by 4x.
-   🎨 **Style Control**: Inject your own direction (e.g., *"Make it cyberpunk with fire"*) and watch the transition adapt.
-   🛡️ **Robust Pipeline**: Includes automatic retries, soft-failure fallbacks (cross-dissolve), and perfect anchor fidelity (using original images).

## 🚀 Quick Start (Colab)

The easiest way to try Sirius is via our interactive notebook:

[**▶️ Open Demo in Google Colab**](https://colab.research.google.com/github/MarsZDF/bfl-hack-sirius/blob/main/demo.ipynb)

## 📦 Installation

```bash
pip install git+https://github.com/MarsZDF/bfl-hack-sirius.git
```

## 🏗 Architecture

Sirius operates as a studio of three AI agents:

1.  **The Director (`director.py`)**: 
    -   *Brain:* Claude 3.5 Sonnet.
    -   *Role:* Analyzes Start/End images. Writes a 16-part prompt sequence that evolves textual concepts linearly (e.g., "Cat" -> "Cat-Dog Hybrid" -> "Dog").
    -   *Logic:* Enforces "Novelist" prompting style for FLUX.

2.  **The Animator (`animator.py`)**: 
    -   *Engine:* FLUX.2 (Pro/Dev/Klein).
    -   *Role:* Executes the prompts. Uses **FLUX.1 [pro]** for high-res details and **FLUX.1 [dev]** for speed.
    -   *Tech:* Uses `asyncio` and `Runware` websockets to generate frames concurrently.

3.  **The Editor (`editor.py`)**: 
    -   *Tool:* FFmpeg / ImageIO.
    -   *Role:* Assembles the frames, handles resizing/padding (no stretching!), and applies framerate logic (8fps for smooth morphs).

## 🛠 Configuration

You need API keys for:
1.  **Anthropic**: `ANTHROPIC_API_KEY`
2.  **Runware** (Recommended) or **BFL**: `RUNWARE_API_KEY`

## 📖 Example

```python
from sirius import morph

# Create a cyberpunk transition for TikTok
result = morph(
    "me.jpg",
    "avatar.jpg",
    user_context="Transform into a neon-lit cyberpunk cyborg",
    output_dir="outputs"
)

print(f"Video saved: {result.video_path}")
```