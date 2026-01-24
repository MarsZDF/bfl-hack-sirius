"""Director module: Combined image analysis and transition planning."""

from typing import Any

from ._anthropic_client import ClaudeClient
from ._config import FLUX_CONSISTENCY_PARAMS, TRANSITION_DESCRIPTIONS
from ._types import ImageAnalysis, TransitionPlan, TransitionStyle
from .exceptions import AnalysisError, AnalysisParseError

# =============================================================================
# Prompt Templates
# =============================================================================

ANALYSIS_PROMPT = """Analyze both images for AI image generation recreation.

For EACH image, extract:
1. Subject: What is the main subject? (front-load this in prompts - FLUX reads left-to-right)
2. Style: Art style (photo, illustration, painting, 3D render, etc.)
3. Lighting: Describe like a photographer (direction, quality, color temperature)
   - Direction: "from the left", "from above", "backlit"
   - Quality: "soft diffused", "harsh direct", "dappled"
   - Temperature: "warm golden", "cool blue", "neutral"
4. Mood: Emotional tone (serene, energetic, mysterious, etc.)
5. Colors: List 3-5 dominant colors as BOTH name AND hex code (e.g., "deep navy #1a365d")
6. Full Prompt: Write a FLUX-optimized prompt using this structure:
   "[Subject doing action]. [Setting/background]. [Key details]. [Lighting description]. [Mood/atmosphere]."
   - Front-load the subject (most important element first)
   - Describe what IS there, never what isn't (no negatives)
   - Use specific color hex codes for precision
7. Summary: One-line summary of the image

Return as JSON with this structure:
{
  "image_a": {
    "subject": "...",
    "style": "...",
    "lighting": "...",
    "mood": "...",
    "colors": ["deep navy #1a365d", "warm gold #d4a574", ...],
    "prompt": "...",
    "summary": "..."
  },
  "image_b": {
    "subject": "...",
    "style": "...",
    "lighting": "...",
    "mood": "...",
    "colors": ["forest green #2d5a27", "cream white #f5f5dc", ...],
    "prompt": "...",
    "summary": "..."
  }
}"""

SCREENWRITER_MASTER_PROMPT = """
You are generating {num_frames} intermediate image descriptions for a smooth 
visual transition from START to END.

=== CONTEXT ===
START IMAGE: {summary_a}
END IMAGE: {summary_b}

START PROMPT (detailed): 
{prompt_a}

END PROMPT (detailed):
{prompt_b}

=== CRITICAL USER INSTRUCTION ===
The user has explicitly requested: "{user_context}"
You MUST incorporate this direction into EVERY prompt.
If the style requested conflicts with the original images, prioritize the User's requested style/elements.

=== TRANSITION STYLE ===
Style: {transition_style}
{transition_instructions}

=== CRITICAL RULES ===

1. PACING: Frame 1 should be a very subtle evolution of the START IMAGE (approx 90% Start, 10% End). 
   Frame {num_frames} should be the final stage just before the END IMAGE (approx 10% Start, 90% End).
   The transformation must be LINEAR and gradual. Frame 1 must preserve the EXACT composition and 
   subject placement of the START IMAGE to ensure a smooth hand-off from the photo.

2. LIGHTING: {lighting_instruction}
   The light source direction MUST remain consistent unless the style explicitly allows shifts.
   Current lighting in START: {lighting_a}
   Current lighting in END: {lighting_b}

3. STYLE CONSISTENCY: Every prompt must include these exact phrases:
   "{style_keywords}"

4. FLUX PROMPTING STYLE (NOVELIST):
   Write like a novelist, not a search engine. FLUX reads prompts left-to-right, so WORD ORDER MATTERS.
   Use this exact structure:
   "[Subject doing action]. [Setting/background]. [Details]. [Lighting]. [Atmosphere]."

   - Subject: THE MOST IMPORTANT ELEMENT. Put it FIRST. Be specific: "A tabby cat with green eyes" not "cat".
   - Setting: Where is it? Background elements that ground the scene.
   - Details: Textures, materials, specific features worth emphasizing.
   - Lighting: Describe like a photographer - this has the BIGGEST visual impact:
     * Direction: "lit from the left", "backlit by window", "overhead lighting"
     * Quality: "soft diffused light", "harsh direct sunlight", "dappled forest light"
     * Color: "warm golden hour glow", "cool blue twilight", "neutral overcast"
   - Atmosphere: Emotional tone (e.g., "serene and peaceful", "tense and ominous").
   - NO negative prompts. Describe what IS there, never what isn't.
   - Use hex codes for precise colors (e.g., "#d4a574 warm gold", "#1a365d deep navy").

5. ADJACENT SIMILARITY: Frame N and Frame N+1 should share at least 70% of 
   their descriptive words. Smooth transitions come from gradual word changes.

=== OUTPUT FORMAT ===
Return a JSON array of exactly {num_frames} strings. No markdown, no explanation.
Each string is a complete prompt ready to send to Flux.

[
  "frame 1 prompt here",
  "frame 2 prompt here",
  ...
]
"""


# =============================================================================
# Director Functions
# =============================================================================


def analyze_images(
    image_a_path: str,
    image_b_path: str,
    client: ClaudeClient | None = None,
) -> tuple[ImageAnalysis, ImageAnalysis]:
    """Analyze both input images using Claude Vision.

    Args:
        image_a_path: Path to first (source) image.
        image_b_path: Path to second (target) image.
        client: Optional Claude client (creates new if not provided).

    Returns:
        Tuple of (analysis_a, analysis_b).

    Raises:
        AnalysisError: If analysis fails.
    """
    if client is None:
        client = ClaudeClient()

    response = client.analyze_image_pair(
        image_a_path,
        image_b_path,
        ANALYSIS_PROMPT,
    )

    return _parse_analysis_response(response)


def plan_transition(
    analysis_a: ImageAnalysis,
    analysis_b: ImageAnalysis,
    frame_count: int = 16,
    style: TransitionStyle | str = TransitionStyle.MORPH,
    client: ClaudeClient | None = None,
    user_context: str | None = None,
) -> TransitionPlan:
    """Generate intermediate prompts for the transition.

    Args:
        analysis_a: Analysis of source image.
        analysis_b: Analysis of target image.
        frame_count: Number of frames to generate.
        style: Transition style to use.
        client: Optional Claude client.

    Returns:
        TransitionPlan with prompts for each frame.

    Raises:
        AnalysisError: If planning fails.
    """
    if client is None:
        client = ClaudeClient()

    # Normalize style
    if isinstance(style, str):
        style = TransitionStyle(style)

    style_description = TRANSITION_DESCRIPTIONS.get(style.value, "")

    # Determine lighting instruction based on consistency params
    lighting_rule = FLUX_CONSISTENCY_PARAMS["interpolated"]["lighting_direction"]["rule"]
    lighting_instruction = (
        f"{lighting_rule} "
        "If they are the same, KEEP IT LOCKED."
    )

    # Determine style keywords
    # For now, we prefer the start style to maintain consistency,
    # unless the style transition is the goal (e.g. fade/surreal).
    # But for FLUX stability, repeating key style descriptors is good.
    style_keywords = analysis_a.style
    if style_keywords != analysis_b.style:
         style_keywords = f"{style_keywords} transitioning to {analysis_b.style}"

    prompt = SCREENWRITER_MASTER_PROMPT.format(
        num_frames=frame_count,
        summary_a=analysis_a.summary,
        summary_b=analysis_b.summary,
        prompt_a=analysis_a.prompt,
        prompt_b=analysis_b.prompt,
        transition_style=style.value,
        transition_instructions=style_description,
        lighting_instruction=lighting_instruction,
        lighting_a=analysis_a.lighting,
        lighting_b=analysis_b.lighting,
        style_keywords=style_keywords,
        user_context=user_context or "No specific user instructions.",
    )

    response_text = client.generate_text(prompt)
    prompts = _parse_transition_response(response_text, frame_count)

    return TransitionPlan(
        prompts=prompts,
        frame_count=frame_count,
        transition_style=style,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )


def describe_images(
    image_a_path: str,
    image_b_path: str,
    client: ClaudeClient | None = None,
) -> str:
    """Analyze images and return a human-readable description of what Claude sees.

    This is useful for debugging and understanding what the model extracts
    from your images before running the full pipeline.

    Args:
        image_a_path: Path to first (source) image.
        image_b_path: Path to second (target) image.
        client: Optional Claude client.

    Returns:
        Human-readable string describing both images.

    Example:
        >>> from sirius import describe_images
        >>> print(describe_images("cat.png", "dog.png"))
    """
    analysis_a, analysis_b = analyze_images(image_a_path, image_b_path, client)

    lines = [
        "=" * 60,
        "IMAGE ANALYSIS",
        "=" * 60,
        "",
        "-" * 60,
        "IMAGE A (Start)",
        "-" * 60,
        analysis_a.describe(),
        "",
        "-" * 60,
        "IMAGE B (End)",
        "-" * 60,
        analysis_b.describe(),
        "=" * 60,
    ]
    return "\n".join(lines)


def direct(
    image_a_path: str,
    image_b_path: str,
    frame_count: int = 16,
    style: TransitionStyle | str = TransitionStyle.MORPH,
    client: ClaudeClient | None = None,
    user_context: str | None = None,
) -> TransitionPlan:
    """Full Director flow: analyze images and plan transition in one call.

    This is the main entry point for Step 1 of the pipeline.

    Args:
        image_a_path: Path to source image.
        image_b_path: Path to target image.
        frame_count: Number of frames to generate.
        style: Transition style to use.
        client: Optional Claude client (reused for both calls).

    Returns:
        Complete TransitionPlan ready for animation.

    Raises:
        AnalysisError: If analysis or planning fails.
    """
    if client is None:
        client = ClaudeClient()

    # Analyze both images
    analysis_a, analysis_b = analyze_images(image_a_path, image_b_path, client)

    # Plan the transition
    plan = plan_transition(
        analysis_a,
        analysis_b,
        frame_count=frame_count,
        style=style,
        client=client,
        user_context=user_context,
    )

    return plan


# =============================================================================
# Response Parsing Helpers
# =============================================================================


def _parse_analysis_response(
    response: dict[str, Any],
) -> tuple[ImageAnalysis, ImageAnalysis]:
    """Parse Claude's analysis response into ImageAnalysis objects."""
    try:
        image_a_data = response.get("image_a", {})
        image_b_data = response.get("image_b", {})

        analysis_a = ImageAnalysis(
            prompt=image_a_data.get("prompt", ""),
            subject=image_a_data.get("subject", ""),
            style=image_a_data.get("style", ""),
            lighting=image_a_data.get("lighting", ""),
            mood=image_a_data.get("mood", ""),
            colors=image_a_data.get("colors", []),
            summary=image_a_data.get("summary", ""),
        )

        analysis_b = ImageAnalysis(
            prompt=image_b_data.get("prompt", ""),
            subject=image_b_data.get("subject", ""),
            style=image_b_data.get("style", ""),
            lighting=image_b_data.get("lighting", ""),
            mood=image_b_data.get("mood", ""),
            colors=image_b_data.get("colors", []),
            summary=image_b_data.get("summary", ""),
        )

        # Validate we got meaningful data
        if not analysis_a.prompt or not analysis_b.prompt:
            raise AnalysisError(
                "Failed to extract prompts from analysis",
                context={"response": response},
            )

        return analysis_a, analysis_b

    except KeyError as e:
        raise AnalysisParseError(str(response), f"Missing key: {e}") from e


def _parse_transition_response(response_text: str, expected_count: int) -> list[str]:
    """Parse Claude's transition planning response."""
    import json

    # Try to extract JSON
    text = response_text.strip()

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

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON list first (as requested in prompt)
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end])
            except json.JSONDecodeError:
                # Fallback to looking for object
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    try:
                        data = json.loads(text[start:end])
                    except json.JSONDecodeError as e:
                        raise AnalysisParseError(response_text, str(e)) from e
                else:
                    raise AnalysisParseError(response_text, "No JSON found")
        else:
            # Fallback to looking for object
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end])
                except json.JSONDecodeError as e:
                    raise AnalysisParseError(response_text, str(e)) from e
            else:
                raise AnalysisParseError(response_text, "No JSON found")

    # Handle both list (direct) and dict (wrapper)
    if isinstance(data, list):
        prompts = data
    elif isinstance(data, dict):
        prompts = data.get("prompts", [])
    else:
        raise AnalysisParseError(response_text, f"Unexpected JSON type: {type(data)}")

    if not isinstance(prompts, list):
        raise AnalysisParseError(response_text, "Parsed data is not a list")

    if len(prompts) != expected_count:
        # Accept close enough - Claude sometimes gives N-1 or N+1
        if abs(len(prompts) - expected_count) > 2:
            raise AnalysisParseError(
                response_text,
                f"Expected {expected_count} prompts, got {len(prompts)}",
            )

    return [str(p) for p in prompts]


# =============================================================================
# "What Changed?" Feature
# =============================================================================

DESCRIBE_CHANGES_PROMPT = """Analyze these two images and describe what changed between them.

Write a concise, engaging description of the transformation as if narrating a timelapse or before/after reveal.

Focus on:
1. **Subject changes**: How did the main subject transform? (age, state, condition)
2. **Environment changes**: What shifted in the background or setting?
3. **Style/mood changes**: Did the lighting, colors, or atmosphere change?
4. **Time implications**: How much time seems to have passed? What story does this tell?

Format your response as:
{
  "headline": "A punchy one-liner for social media (max 10 words)",
  "description": "2-3 sentence narrative description of the transformation",
  "changes": [
    "Specific change 1",
    "Specific change 2",
    "Specific change 3"
  ],
  "time_passed": "Estimated time between images (e.g., '50 years', 'one season', 'moments')",
  "mood_shift": "From [start mood] to [end mood]"
}"""


def describe_changes(
    image_a_path: str,
    image_b_path: str,
    client: ClaudeClient | None = None,
) -> dict[str, Any]:
    """Analyze two images and describe what changed between them.

    This is the "What Changed?" feature - great for demos and social sharing.
    Shows Claude's understanding of the transformation.

    Args:
        image_a_path: Path to first (before) image.
        image_b_path: Path to second (after) image.
        client: Optional Claude client.

    Returns:
        Dictionary with headline, description, changes list, time_passed, mood_shift.

    Example:
        >>> from sirius import describe_changes
        >>> result = describe_changes("young.jpg", "old.jpg")
        >>> print(result["headline"])
        "A lifetime in two frames"
        >>> print(result["description"])
        "A young woman with bright eyes transforms into her elderly self..."
    """
    if client is None:
        client = ClaudeClient()

    response = client.analyze_image_pair(
        image_a_path,
        image_b_path,
        DESCRIBE_CHANGES_PROMPT,
    )

    # Ensure we have the expected fields with defaults
    return {
        "headline": response.get("headline", "A transformation"),
        "description": response.get("description", ""),
        "changes": response.get("changes", []),
        "time_passed": response.get("time_passed", "unknown"),
        "mood_shift": response.get("mood_shift", ""),
    }
