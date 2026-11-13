import pytest
import torch

from src.utils.device import DeviceManager


@pytest.fixture
def hw(monkeypatch):
    """Fake which accelerators exist: hw(cuda=..., mps=...)."""
    def set_hw(cuda=False, mps=False):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    return set_hw


@pytest.mark.parametrize(
    "cuda, mps, device, dtype",
    [
        (True, True, "cuda", torch.bfloat16),
        (False, True, "mps", torch.bfloat16),
        (False, False, "cpu", torch.float32),
    ],
)
def test_auto_detect_priority_and_dtype(hw, cuda, mps, device, dtype):
    hw(cuda=cuda, mps=mps)
    dm = DeviceManager.detect()
    assert (dm.device, dm.dtype) == (device, dtype)
    assert DeviceManager.detect(None) == dm


def test_explicit_device(hw):
    hw(cuda=False, mps=True)
    assert DeviceManager.detect("cpu").device == "cpu"
    assert DeviceManager.detect("mps").dtype == torch.bfloat16
    with pytest.raises(RuntimeError, match="not available"):
        DeviceManager.detect("cuda")
    with pytest.raises(ValueError, match="Unknown device"):
        DeviceManager.detect("tpu")


def test_real_machine_detects_something():
    dm = DeviceManager.detect()
    assert torch.zeros(1, device=dm.device, dtype=dm.dtype).device.type == dm.device
