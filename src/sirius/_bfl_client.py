"""BFL FLUX.2 API client with retry logic."""

import time
from io import BytesIO
from typing import Any

import requests
from PIL import Image

from ._config import (
    BFL_API_BASE,
    BFL_MAX_RETRIES,
    BFL_POLL_INTERVAL,
    BFL_POLL_TIMEOUT,
    FLUX_KLEIN,
    get_bfl_api_key,
)
from .exceptions import (
    BFLAPIError,
    GenerationTimeoutError,
    get_retry_delay,
)


class BFLClient:
    """Client for BFL FLUX.2 API with retry logic and polling."""

    # Endpoints that support guidance and steps parameters
    ENDPOINTS_WITH_GUIDANCE = {"flux-dev"}

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize BFL client.

        Args:
            api_key: BFL API key. If not provided, reads from BFL_API_KEY env var.
        """
        self.api_key = api_key or get_bfl_api_key()
        self.base_url = BFL_API_BASE

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with auth."""
        return {
            "accept": "application/json",
            "x-key": self.api_key,
            "Content-Type": "application/json",
        }

    def _build_payload(
        self,
        prompt: str,
        model: str,
        seed: int,
        width: int,
        height: int,
        guidance: float | None = None,
        steps: int | None = None,
    ) -> dict[str, Any]:
        """Build API payload based on endpoint capabilities.

        Different FLUX endpoints support different parameters:
        - flux-dev: supports guidance_scale, num_inference_steps
        - flux-2-klein-9b, flux-pro-1.1: only seed, width, height
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "seed": seed,
            "width": width,
            "height": height,
        }

        # Only add guidance/steps for endpoints that support them
        if model in self.ENDPOINTS_WITH_GUIDANCE:
            if guidance is not None:
                payload["guidance_scale"] = guidance
            if steps is not None:
                payload["num_inference_steps"] = steps

        return payload

    def _poll_result(
        self,
        polling_url: str,
        timeout: float = BFL_POLL_TIMEOUT,
    ) -> dict[str, Any]:
        """Poll for async generation result.

        Args:
            polling_url: URL to poll for status.
            timeout: Maximum seconds to wait.

        Returns:
            Result dict with image URL.

        Raises:
            GenerationTimeoutError: If polling times out.
            BFLAPIError: If generation fails.
        """
        start_time = time.time()
        headers = {"accept": "application/json", "x-key": self.api_key}

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise GenerationTimeoutError(timeout, polling_url)

            time.sleep(BFL_POLL_INTERVAL)

            response = requests.get(polling_url, headers=headers, timeout=30)
            response.raise_for_status()
            status = response.json()

            if status.get("status") == "Ready":
                return status
            elif status.get("status") in ("Error", "Failed"):
                raise BFLAPIError(
                    f"Generation failed: {status.get('error', 'Unknown error')}",
                    endpoint=polling_url,
                    retryable=False,
                )

            # Log progress if available
            progress = status.get("progress")
            if progress is not None:
                # Progress available but we don't print here - let caller handle
                pass

    def generate(
        self,
        prompt: str,
        *,
        model: str = FLUX_KLEIN,
        seed: int = 42,
        width: int = 1024,
        height: int = 576,
        guidance: float | None = None,
        steps: int | None = None,
        timeout: float = BFL_POLL_TIMEOUT,
    ) -> tuple[Image.Image, int]:
        """Generate an image using FLUX.2.

        Args:
            prompt: Text prompt for generation.
            model: FLUX model to use.
            seed: Random seed for reproducibility.
            width: Output width in pixels.
            height: Output height in pixels.
            guidance: Guidance scale (only for flux-dev).
            steps: Inference steps (only for flux-dev).
            timeout: Max seconds to wait for generation.

        Returns:
            Tuple of (PIL Image, duration_ms).

        Raises:
            BFLAPIError: If API call fails.
            GenerationTimeoutError: If generation times out.
        """
        endpoint = f"{self.base_url}/{model}"
        payload = self._build_payload(
            prompt=prompt,
            model=model,
            seed=seed,
            width=width,
            height=height,
            guidance=guidance,
            steps=steps,
        )

        start_time = time.time()
        last_error: Exception | None = None

        for attempt in range(1, BFL_MAX_RETRIES + 1):
            try:
                # Submit generation request
                response = requests.post(
                    endpoint,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=30,
                )

                if response.status_code == 422:
                    # Unprocessable entity - likely invalid parameters
                    raise BFLAPIError(
                        f"Invalid parameters for {model}: {response.text}",
                        status_code=422,
                        endpoint=endpoint,
                        retryable=False,
                    )

                response.raise_for_status()
                result = response.json()

                # Get polling URL
                polling_url = result.get("polling_url")
                if not polling_url:
                    raise BFLAPIError(
                        f"No polling_url in response: {result}",
                        endpoint=endpoint,
                        retryable=False,
                    )

                # Poll for result
                final_result = self._poll_result(polling_url, timeout)

                # Download image
                if "result" in final_result and "sample" in final_result["result"]:
                    image_url = final_result["result"]["sample"]
                    image_response = requests.get(image_url, timeout=60)
                    image_response.raise_for_status()
                    image = Image.open(BytesIO(image_response.content))
                else:
                    raise BFLAPIError(
                        f"Unexpected response format: {final_result.keys()}",
                        endpoint=endpoint,
                        retryable=False,
                    )

                duration_ms = int((time.time() - start_time) * 1000)
                return image, duration_ms

            except requests.exceptions.RequestException as e:
                # Network error - retryable
                last_error = BFLAPIError(
                    f"Network error: {e}",
                    endpoint=endpoint,
                    retryable=True,
                )
                if attempt < BFL_MAX_RETRIES:
                    delay = get_retry_delay(last_error, attempt)
                    time.sleep(delay)
                continue

            except BFLAPIError:
                raise

            except Exception as e:
                raise BFLAPIError(
                    f"Unexpected error: {e}",
                    endpoint=endpoint,
                    retryable=False,
                ) from e

        # All retries exhausted
        if last_error:
            raise last_error
        raise BFLAPIError("Generation failed after all retries", endpoint=endpoint)

    def generate_batch(
        self,
        prompts: list[str],
        *,
        model: str = FLUX_KLEIN,
        seed: int = 42,
        width: int = 1024,
        height: int = 576,
        **kwargs: Any,
    ) -> list[tuple[Image.Image, int]]:
        """Generate multiple images sequentially.

        Note: BFL API doesn't support true batch inference, so this is
        sequential. Use parallel generation in animator.py for concurrency.

        Args:
            prompts: List of prompts to generate.
            model: FLUX model to use.
            seed: Random seed (same for all for consistency).
            width: Output width.
            height: Output height.
            **kwargs: Additional args passed to generate().

        Returns:
            List of (PIL Image, duration_ms) tuples.
        """
        results = []
        for prompt in prompts:
            result = self.generate(
                prompt=prompt,
                model=model,
                seed=seed,
                width=width,
                height=height,
                **kwargs,
            )
            results.append(result)
        return results
