"""Runware API client for FLUX models."""

import asyncio
import threading
import time
from io import BytesIO
from typing import Any

import requests
from PIL import Image
from runware import Runware, IImageInference

from ._config import (
    RUNWARE_FLUX_PRO,
    get_runware_api_key,
)
from .exceptions import (
    GenerationError,
    GenerationTimeoutError,
)

# Thread-local storage for event loops and clients
_thread_local = threading.local()

# Enable nested event loops (needed for Jupyter/Colab)
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass


def _get_thread_loop() -> asyncio.AbstractEventLoop:
    """Get or create an event loop for the current thread."""
    try:
        # Try to get the running loop first (works in Jupyter/Colab)
        loop = asyncio.get_running_loop()
        return loop
    except RuntimeError:
        pass

    # No running loop - create or get one for this thread
    if not hasattr(_thread_local, 'loop') or _thread_local.loop is None or _thread_local.loop.is_closed():
        _thread_local.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_thread_local.loop)
    return _thread_local.loop


class RunwareClient:
    """Client for Runware API with synchronous wrapper."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize Runware client.

        Args:
            api_key: Runware API key. If not provided, reads from RUNWARE_API_KEY env var.
        """
        self.api_key = api_key or get_runware_api_key()
        if not self.api_key:
            raise ValueError("RUNWARE_API_KEY not set")

    async def _generate_async(
        self,
        prompt: str,
        model: str,
        seed: int,
        width: int,
        height: int,
        steps: int | None = None,
        guidance: float | None = None,
    ) -> tuple[Image.Image, int]:
        """Internal async generation method."""
        start_time = time.time()

        # Always create a new Runware instance to avoid state pollution
        runware = Runware(api_key=self.api_key)

        try:
            await runware.connect()

            # Build request parameters dynamically
            params = {
                "positivePrompt": prompt,
                "model": model,
                "seed": seed,
                "width": width,
                "height": height,
                "outputType": "URL",
                "outputFormat": "png",
            }

            # Only add CFG/Steps for models that support it
            if "100@1" in model:
                params["steps"] = steps or 30
                params["CFGScale"] = guidance or 3.5
            
            request = IImageInference(**params)

            images = await runware.imageInference(requestImage=request)
            
            # Handle response structure
            if isinstance(images, list):
                results = images
            elif hasattr(images, 'results'):
                results = images.results
            else:
                results = [images] if images else []

            if not results:
                raise GenerationError(f"Runware returned no results for prompt: {prompt[:50]}...")
            
            result = results[0]
            
            url = getattr(result, 'imageURL', None) or getattr(result, 'url', None)
            if not url and isinstance(result, dict):
                url = result.get('imageURL') or result.get('url')

            if not url:
                raise GenerationError(f"Runware result missing imageURL: {result}")

            # Download image
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            
            duration_ms = int((time.time() - start_time) * 1000)
            return image, duration_ms

        except Exception as e:
            raise GenerationError(f"Runware generation failed: {e}") from e
        finally:
            # Always disconnect to clean up websockets
            try:
                if runware.connected:
                    await runware.disconnect()
            except Exception:
                pass

    def generate(
        self,
        prompt: str,
        *,
        model: str = RUNWARE_FLUX_PRO,
        seed: int = 42,
        width: int = 1024,
        height: int = 576,
        guidance: float | None = None,
        steps: int | None = None,
        timeout: float = 120.0,
    ) -> tuple[Image.Image, int]:
        """Generate an image using Runware (synchronous wrapper)."""
        
        # Get thread-local event loop (or running loop)
        loop = _get_thread_loop()

        # If loop is running (e.g. Colab), nest_asyncio allows run_until_complete
        # If loop is new (thread), run_until_complete works standardly
        return loop.run_until_complete(
            self._generate_async(
                prompt=prompt,
                model=model,
                seed=seed,
                width=width,
                height=height,
                steps=steps,
                guidance=guidance,
            )
        )