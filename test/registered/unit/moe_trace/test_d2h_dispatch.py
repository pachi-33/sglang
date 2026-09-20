import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.utils import common as utils


class _FakeTensor:
    def __init__(self, device_type: str):
        self.device = SimpleNamespace(type=device_type)
        # Simulates torch_npu's compatibility alias without requiring torch_npu.
        self.is_cuda = True
        self.shape = (2, 3)
        self.dtype = torch.float32
        self.to = Mock()
        self.record_stream = Mock()


class TestAsyncD2HDeviceDispatch(unittest.TestCase):
    def test_npu_alias_uses_plain_copy_without_cuda_apis(self):
        source = _FakeTensor("npu")
        host = object()
        source.to.return_value = host

        with (
            patch.object(utils.torch, "empty") as empty,
            patch.object(utils.torch, "get_device_module") as get_device_module,
            patch.object(utils, "is_pin_memory_available", return_value=False),
        ):
            result = utils.async_d2h(source)

        self.assertIs(result, host)
        source.to.assert_called_once_with("cpu")
        empty.assert_not_called()
        get_device_module.assert_not_called()
        source.record_stream.assert_not_called()

    def test_cuda_keeps_pinned_copy_and_stream_lifetime(self):
        source = _FakeTensor("cuda")
        host = Mock()
        stream = object()
        device_module = SimpleNamespace(current_stream=Mock(return_value=stream))

        with (
            patch.object(utils.torch, "empty", return_value=host) as empty,
            patch.object(
                utils.torch, "get_device_module", return_value=device_module
            ) as get_device_module,
            patch.object(utils, "is_pin_memory_available", return_value=True),
        ):
            result = utils.async_d2h(source)

        self.assertIs(result, host)
        empty.assert_called_once_with(source.shape, dtype=source.dtype, pin_memory=True)
        host.copy_.assert_called_once_with(source, non_blocking=True)
        get_device_module.assert_called_once_with(source.device)
        device_module.current_stream.assert_called_once_with()
        source.record_stream.assert_called_once_with(stream)
        source.to.assert_not_called()
