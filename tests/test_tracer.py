import pytest

import torchtrail


def test_nested_trace_error():
    with torchtrail.trace():
        with pytest.raises(RuntimeError):
            with torchtrail.trace():
                pass

