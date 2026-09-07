import fcntl
import os
import unittest

import torch

V100_UUID = "GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96"
RTX_4070_SUPER_UUID = "GPU-75341d61-b0b3-969b-8ef8-4b750d11ade4"

# GPU unit tests are intentionally constrained to the two devices whose kernel
# contracts are validated by this repository.  UUID and capability are paired:
# accepting either independently would allow a different GPU to silently run a
# test under an unsupported Triton target.
SUPPORTED_GPU_CONTRACTS = {
    V100_UUID: (7, 0),
    RTX_4070_SUPER_UUID: (8, 9),
}


def gpu_test_lock_path(uuid: str) -> str:
    """Return the one serialization lock shared by all work on a GPU UUID."""
    capability = SUPPORTED_GPU_CONTRACTS[uuid]
    return f"/tmp/qwen35-gpu-{uuid}-sm{capability[0]}{capability[1]}.lock"


V100_GPU_LOCK_PATH = gpu_test_lock_path(V100_UUID)


class Qwen35GPUCompatTestCase(unittest.TestCase):
    """Base class for serialized GPU compatibility tests on validated GPUs."""

    _lock = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA unavailable")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        visible_uuids = [entry.strip() for entry in visible.split(",") if entry.strip()]
        if len(visible_uuids) != 1:
            raise unittest.SkipTest(
                "set CUDA_VISIBLE_DEVICES to exactly one validated GPU UUID"
            )
        uuid = visible_uuids[0]
        expected_capability = SUPPORTED_GPU_CONTRACTS.get(uuid)
        if expected_capability is None:
            raise unittest.SkipTest(
                "set CUDA_VISIBLE_DEVICES to a validated SM70 V100 or SM89 RTX 4070 SUPER UUID"
            )
        actual_capability = torch.cuda.get_device_capability()
        if actual_capability != expected_capability:
            raise AssertionError(
                f"GPU contract mismatch for {uuid}: expected {expected_capability}, "
                f"got {actual_capability}"
            )
        cls._lock = open(gpu_test_lock_path(uuid), "a+")
        fcntl.flock(cls._lock.fileno(), fcntl.LOCK_EX)

    @classmethod
    def tearDownClass(cls):
        if cls._lock is not None:
            fcntl.flock(cls._lock.fileno(), fcntl.LOCK_UN)
            cls._lock.close()
        super().tearDownClass()


# Kept as a compatibility alias for the existing unit and integration imports.
# It now accepts either explicitly validated device contract above.
V100TestCase = Qwen35GPUCompatTestCase


class TestEnvironment(unittest.TestCase):
    def test_required_versions(self):
        self.assertTrue(torch.__version__.startswith("2.3."), torch.__version__)
        import triton

        self.assertTrue(triton.__version__.startswith("2.3."), triton.__version__)
