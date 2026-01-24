"""Tests for Director module."""

from unittest.mock import MagicMock, patch

import pytest
from sirius.director import analyze_images, plan_transition
from sirius._types import ImageAnalysis, TransitionStyle

@pytest.fixture
def mock_claude_client():
    client = MagicMock()
    
    # Mock analyze response
    client.analyze_image_pair.return_value = {
        "image_a": {
            "prompt": "A prompt",
            "subject": "A subject",
            "style": "Photo",
            "summary": "Summary A"
        },
        "image_b": {
            "prompt": "B prompt",
            "subject": "B subject",
            "style": "Photo",
            "summary": "Summary B"
        }
    }
    
    # Mock plan response
    client.generate_text.return_value = """
    {
        "prompts": ["P1", "P2", "P3", "P4"]
    }
    """
    
    return client

def test_analyze_images(mock_claude_client):
    """Test image analysis."""
    a, b = analyze_images("path/a.png", "path/b.png", client=mock_claude_client)
    
    assert isinstance(a, ImageAnalysis)
    assert a.prompt == "A prompt"
    assert b.summary == "Summary B"
    mock_claude_client.analyze_image_pair.assert_called_once()

def test_plan_transition(mock_claude_client):
    """Test transition planning."""
    analysis_a = ImageAnalysis(
        prompt="A", subject="A", style="S", lighting="L", 
        mood="M", colors=[], summary="Sum A"
    )
    analysis_b = ImageAnalysis(
        prompt="B", subject="B", style="S", lighting="L", 
        mood="M", colors=[], summary="Sum B"
    )
    
    plan = plan_transition(
        analysis_a, 
        analysis_b, 
        frame_count=4, 
        style=TransitionStyle.MORPH,
        client=mock_claude_client
    )
    
    assert len(plan.prompts) == 4
    assert plan.prompts[0] == "P1"
    assert plan.transition_style == TransitionStyle.MORPH
    mock_claude_client.generate_text.assert_called_once()
