"""Anthropic Claude API client for vision and text."""

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import anthropic
from PIL import Image

from ._config import CLAUDE_MODEL, get_anthropic_api_key
from .exceptions import AnalysisParseError, ImageLoadError, VisionAPIError


class ClaudeClient:
    """Client for Anthropic Claude API (Vision + Text)."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize Claude client.

        Args:
            api_key: Anthropic API key. If not provided, reads from env var.
        """
        self.api_key = api_key or get_anthropic_api_key()
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = CLAUDE_MODEL

    def _load_image_as_base64(self, image_path: str) -> tuple[str, str]:
        """Load image and convert to base64.

        Args:
            image_path: Path to image file.

        Returns:
            Tuple of (base64_data, media_type).

        Raises:
            ImageLoadError: If image cannot be loaded.
        """
        path = Path(image_path)
        if not path.exists():
            raise ImageLoadError(image_path, "File not found")

        # Determine media type
        suffix = path.suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_types.get(suffix)
        if not media_type:
            raise ImageLoadError(image_path, f"Unsupported format: {suffix}")

        try:
            with open(path, "rb") as f:
                image_bytes = f.read()

            # Resize if too large (> 4MB) or dimensions > 1568px (Claude optimal)
            if len(image_bytes) > 4 * 1024 * 1024:
                img = Image.open(BytesIO(image_bytes))
                
                # Calculate new size maintaining aspect ratio
                max_dim = 1568
                if max(img.size) > max_dim:
                    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                
                # Convert to RGB if necessary (e.g. RGBA to JPEG)
                if img.mode in ('RGBA', 'P') and suffix in ['.jpg', '.jpeg']:
                    img = img.convert('RGB')
                
                # Save to buffer
                buffer = BytesIO()
                # Use original format if possible, else JPEG for compression
                save_format = "JPEG" if suffix in ['.jpg', '.jpeg'] else (img.format or "PNG")
                img.save(buffer, format=save_format, quality=85, optimize=True)
                image_bytes = buffer.getvalue()

            image_data = base64.standard_b64encode(image_bytes).decode("utf-8")
            return image_data, media_type
        except Exception as e:
            raise ImageLoadError(image_path, str(e)) from e

    def analyze_image(self, image_path: str, prompt: str) -> dict[str, Any]:
        """Send image to Claude Vision for analysis.

        Args:
            image_path: Path to image file.
            prompt: Analysis prompt.

        Returns:
            Parsed JSON response.

        Raises:
            ImageLoadError: If image cannot be loaded.
            VisionAPIError: If API call fails.
            AnalysisParseError: If response cannot be parsed.
        """
        image_data, media_type = self._load_image_as_base64(image_path)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
            )
        except anthropic.APIError as e:
            raise VisionAPIError(
                f"Claude Vision API error: {e}",
                status_code=getattr(e, "status_code", None),
                retryable=getattr(e, "status_code", 0) >= 500,
            ) from e

        # Extract text content
        text_content = ""
        for block in response.content:
            if block.type == "text":
                text_content = block.text
                break

        # Parse JSON from response
        return self._parse_json_response(text_content)

    def analyze_single_image(self, image_path: str, prompt: str) -> dict[str, Any]:
        """Analyze a single image (alias for analyze_image).

        Args:
            image_path: Path to image file.
            prompt: Analysis prompt.

        Returns:
            Parsed JSON response.
        """
        return self.analyze_image(image_path, prompt)

    def analyze_image_pair(
        self,
        image_a_path: str,
        image_b_path: str,
        prompt: str,
    ) -> dict[str, Any]:
        """Analyze both images in a single API call.

        Args:
            image_a_path: Path to first image.
            image_b_path: Path to second image.
            prompt: Analysis prompt.

        Returns:
            Parsed JSON response.
        """
        image_a_data, media_type_a = self._load_image_as_base64(image_a_path)
        image_b_data, media_type_b = self._load_image_as_base64(image_b_path)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "IMAGE A (Start):",
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type_a,
                                    "data": image_a_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": "IMAGE B (End):",
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type_b,
                                    "data": image_b_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
            )
        except anthropic.APIError as e:
            raise VisionAPIError(
                f"Claude Vision API error: {e}",
                status_code=getattr(e, "status_code", None),
                retryable=getattr(e, "status_code", 0) >= 500,
            ) from e

        # Extract text content
        text_content = ""
        for block in response.content:
            if block.type == "text":
                text_content = block.text
                break

        return self._parse_json_response(text_content)

    def generate_text(self, prompt: str) -> str:
        """Generate text completion.

        Args:
            prompt: Text prompt.

        Returns:
            Generated text.
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
        except anthropic.APIError as e:
            raise VisionAPIError(
                f"Claude API error: {e}",
                status_code=getattr(e, "status_code", None),
                retryable=getattr(e, "status_code", 0) >= 500,
            ) from e

        for block in response.content:
            if block.type == "text":
                return block.text

        return ""

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Parse JSON from Claude response.

        Handles responses that may have markdown code blocks.

        Args:
            text: Raw response text.

        Returns:
            Parsed JSON as dict.

        Raises:
            AnalysisParseError: If JSON cannot be parsed.
        """
        # Try to extract JSON from code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()

        # Try direct parse
        try:
            return json.loads(text)  # type: ignore[no-any-return]
        except json.JSONDecodeError as e:
            # Try to find JSON object in text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])  # type: ignore[no-any-return]
                except json.JSONDecodeError:
                    pass

            raise AnalysisParseError(text, str(e)) from e
