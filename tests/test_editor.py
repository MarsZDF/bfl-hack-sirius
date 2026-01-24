"""Tests for Editor module."""

from unittest.mock import MagicMock, patch
from sirius.editor import assemble_video
from sirius._types import Frame

@patch("sirius.editor.iio")
def test_assemble_video(mock_iio, tmp_path):
    """Test video assembly logic."""
    # Create dummy frames
    frames = []
    for i in range(5):
        f = Frame(
            index=i,
            prompt=f"p{i}",
            path=str(tmp_path / f"f{i}.png"),
            duration_ms=100
        )
        # Create dummy file
        with open(f.path, "w") as fp:
            fp.write("dummy")
        frames.append(f)
        
    output_path = str(tmp_path / "out.mp4")
    
    # Mock image reading
    mock_iio.imread.return_value = MagicMock()
    
    result = assemble_video(frames, output_path)
    
    assert result == output_path
    mock_iio.imwrite.assert_called_once()
    assert mock_iio.imread.call_count == 5
