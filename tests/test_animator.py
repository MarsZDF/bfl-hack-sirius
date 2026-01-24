"""Tests for Animator module."""

from unittest.mock import MagicMock, patch
from PIL import Image

import pytest
from sirius.animator import generate_frames_parallel, animate
from sirius._types import GenerationConfig, TransitionPlan, TransitionStyle, ImageAnalysis

@pytest.fixture
def mock_bfl_client():
    client = MagicMock()
    # Mock generate returning (Image, duration)
    client.generate.return_value = (Image.new("RGB", (100, 100)), 1000)
    return client

def test_generate_frames_parallel(mock_bfl_client, tmp_path):
    """Test parallel frame generation."""
    config = GenerationConfig(workers=2)
    prompts = ["p1", "p2", "p3"]
    output_dir = tmp_path / "frames"
    
    frames = generate_frames_parallel(
        prompts, 
        config, 
        str(output_dir), 
        client=mock_bfl_client
    )
    
    assert len(frames) == 3
    assert frames[0].prompt == "p1"
    assert frames[0].index == 0
    assert mock_bfl_client.generate.call_count == 3

def test_animate_full_flow(mock_bfl_client, tmp_path):
    """Test full animation flow with anchors."""
    plan = TransitionPlan(
        prompts=["start", "mid1", "mid2", "end"],
        frame_count=4,
        transition_style=TransitionStyle.MORPH,
        analysis_a=MagicMock(),
        analysis_b=MagicMock()
    )
    
    frames = animate(
        plan,
        output_dir=str(tmp_path),
        client=mock_bfl_client
    )
    
    assert len(frames) == 4
    # Check that anchors used different models if verified
    # (Implementation detail: animate calls generate with different models)
    assert mock_bfl_client.generate.call_count == 4
