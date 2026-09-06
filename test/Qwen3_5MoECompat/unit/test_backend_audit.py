import unittest
from test.Qwen3_5MoECompat.bench.backend_audit import classify_cuda_event


class TestBackendAuditClassifier(unittest.TestCase):
    def test_accepts_compiled_triton_and_separates_memory(self):
        entries = {"route_combine_fused_kernel_0d1d", "_qk"}
        self.assertEqual(
            classify_cuda_event("route_combine_fused_kernel_0d1d", entries), "compute"
        )
        self.assertEqual(classify_cuda_event("[CUDA memset]", entries), "memory")

    def test_rejects_framework_and_unknown_compute(self):
        entries = {"_qk"}
        for name in (
            "void at::native::vectorized_elementwise_kernel",
            "cublasGemmEx",
            "unknown_kernel",
            "route_combine_fused_kernel_other_specialization",
        ):
            with self.subTest(name=name), self.assertRaises(AssertionError):
                classify_cuda_event(name, entries)
