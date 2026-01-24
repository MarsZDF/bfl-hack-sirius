"""Tests for Pipeline orchestration."""

from unittest.mock import MagicMock, patch
from sirius.pipeline import morph
from sirius._types import MorphResult

@patch("sirius.pipeline.ClaudeClient")
@patch("sirius.pipeline.BFLClient")
@patch("sirius.pipeline.direct")
@patch("sirius.pipeline.animate")
@patch("sirius.pipeline.edit")
@patch("sirius.pipeline.gallium")
def test_morph_pipeline(
    mock_gallium,
    mock_edit,
    mock_animate,
    mock_direct,
    mock_bfl,
    mock_claude,
    tmp_path
):
    """Test the full morph pipeline orchestration."""
    # Setup mocks
    mock_direct.return_value = MagicMock()  # Plan
    
    frame = MagicMock()
    frame.prompt = "test"
    frame.seed = 42
    frame.path = "path/to/frame.png"
    frame.model = "flux"
    frame.duration_ms = 100
    frame.index = 0
    
    mock_animate.return_value = [frame] * 4  # Frames
    mock_edit.return_value = "video.mp4"
    
    result = morph(
        "a.png", 
        "b.png", 
        frame_count=4, 
        output_dir=str(tmp_path),
        track_with_gallium=True
    )
    
    assert isinstance(result, MorphResult)
    assert result.video_path.endswith("video.mp4")
    
    mock_direct.assert_called_once()
    mock_animate.assert_called_once()
    mock_edit.assert_called_once()
    mock_gallium.log.assert_called()
