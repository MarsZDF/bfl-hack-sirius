"""Runware API client for FLUX models."""

import asyncio
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
        
        self.runware = Runware(api_key=self.api_key)

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
        
        if not self.runware.connected:
            await self.runware.connect()

        try:
            request = IImageInference(
                positivePrompt=prompt,
                model=model,
                seed=seed,
                width=width,
                height=height,
                steps=steps or 30,
                CFGScale=guidance or 3.5,
                outputType="URL",
                outputFormat="png"
            )
            
            images = await self.runware.imageInference(requestImage=request)
            
            if not images or not images.results:
                raise GenerationError(f"Runware returned no results for prompt: {prompt[:50]}...")
            
            result = images.results[0]
            if not result.url:
                raise GenerationError("Runware result missing URL")

            # Download image
            response = requests.get(result.url, timeout=60)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            
            duration_ms = int((time.time() - start_time) * 1000)
            return image, duration_ms

        except Exception as e:
            raise GenerationError(f"Runware generation failed: {e}") from e

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
        
        # We need a new event loop if we are in a thread (Animator uses ThreadPoolExecutor)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

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
