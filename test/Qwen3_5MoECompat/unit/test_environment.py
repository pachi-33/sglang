import fcntl
import os
import unittest

import torch

V100_UUID = "GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96"


class V100TestCase(unittest.TestCase):
    """Base class for serialized GPU compatibility tests."""

    _lock = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA unavailable")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if V100_UUID not in visible:
            raise unittest.SkipTest(
                "set CUDA_VISIBLE_DEVICES to the required V100 UUID"
            )
        cls._lock = open("/tmp/qwen35-v100-gpu.lock", "a+")
        fcntl.flock(cls._lock.fileno(), fcntl.LOCK_EX)
        if torch.cuda.get_device_capability() != (7, 0):
            fcntl.flock(cls._lock.fileno(), fcntl.LOCK_UN)
            cls._lock.close()
            raise AssertionError(
                f"expected SM70, got {torch.cuda.get_device_capability()}"
            )

    @classmethod
    def tearDownClass(cls):
        if cls._lock is not None:
            fcntl.flock(cls._lock.fileno(), fcntl.LOCK_UN)
            cls._lock.close()
        super().tearDownClass()


class TestEnvironment(unittest.TestCase):
    def test_required_versions(self):
        self.assertTrue(torch.__version__.startswith("2.3."), torch.__version__)
        import triton

        self.assertTrue(triton.__version__.startswith("2.3."), triton.__version__)
