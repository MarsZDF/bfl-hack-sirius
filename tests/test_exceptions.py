"""Tests for exception hierarchy."""

from sirius.exceptions import (
    AnalysisError,
    BFLAPIError,
    GenerationError,
    PipelineError,
    SiriusError,
)

def test_exception_hierarchy():
    """Test that exceptions inherit correctly."""
    err = BFLAPIError("API failed")
    assert isinstance(err, GenerationError)
    assert isinstance(err, SiriusError)
    assert isinstance(err, Exception)

def test_exception_context():
    """Test that context is formatted in string representation."""
    err = AnalysisError("Failed", context={"code": 500, "details": "server error"})
    msg = str(err)
    assert "Failed" in msg
    assert "code=500" in msg
    assert "details=server error" in msg

def test_pipeline_error():
    """Test pipeline error."""
    err = PipelineError("Something broke", context={"stage": "planning"})
    assert str(err) == "Something broke (stage=planning)"
