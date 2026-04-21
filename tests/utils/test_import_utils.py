import pytest
from diffsynth_engine.utils.import_utils import is_npu_available


class TestIsNpuAvailable:
    def test_is_npu_available_returns_bool(self):
        """Verify is_npu_available returns a boolean."""
        result = is_npu_available()
        assert isinstance(result, bool)

    def test_is_npu_available_callable(self):
        """Verify is_npu_available is callable."""
        assert callable(is_npu_available)
