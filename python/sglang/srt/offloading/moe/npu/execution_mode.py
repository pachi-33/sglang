"""Lifecycle strategies for cached serving and a dedicated prefill worker.

The offloader owns weight storage and the shared layer-prefetch pipeline. A
strategy decides which execution resources to create and what happens at a
forward boundary. Cached serving keeps its existing implementation; prefill
never constructs decode/cache state just to satisfy that implementation.
"""

import logging
from contextlib import contextmanager
from pathlib import Path

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec, get_parallel, get_spec

from ..periodic_lfu import PeriodicLFUConfig, PeriodicLFUPolicy
from ..resources import ResourceKind

logger = logging.getLogger(__name__)


class CachedExecution:
    """Existing colocated prefill/decode with a persistent Hot cache."""

    @staticmethod
    def default_execution_slots(local_experts):
        return local_experts

    def validate_topology(self, owner):
        parallel = get_parallel()
        if parallel.pp_size != 1 or parallel.attn_cp_size != 1:
            raise ValueError("Mixed expert offload requires PP=1 and attention CP=1")

    def validate_quantization(self, owner, quant_spec):
        pass

    def configure(self, owner, path):
        owner.hot_slots = int(owner.catalog.num_experts * owner.options.cache_ratio)
        if owner.hot_slots < 1:
            raise ValueError("Mixed offload requires a nonempty Hot cache")
        if owner.hot_slots > 1280 or owner.catalog.num_experts > 8192:
            raise ValueError(
                "Cache maintenance supports at most 1280 Hot slots and 8192 expert keys per rank"
            )
        cfg = owner.cfg
        profile = cfg.get("offline_stats_path")
        if profile and not Path(profile).is_absolute():
            profile = str(Path(path).resolve().parent / profile)
        owner.policy = PeriodicLFUPolicy(
            PeriodicLFUConfig(
                offline_stats_path=profile,
                **{
                    key: cfg[key]
                    for key in (
                        "update_interval",
                        "min_usage",
                        "replacement_min_gain",
                        "max_swaps",
                        "usage_decay",
                    )
                    if key in cfg
                },
            ),
            owner.catalog,
        )
        owner.initial_keys = owner.policy.initial_keys(
            owner.hot_slots, cfg.get("offline_model_id", owner.model_id)
        )

    def post_init(self, owner):
        owner._init_cached_runtime()

    def forward_scope(self, owner, batch):
        return owner._cached_forward_scope(batch)

    def prepare_decode_weights(self, owner, layer_id, runner_input):
        return owner._prepare_fused(layer_id, runner_input)


class PrefillOnlyExecution:
    """Two layer banks, no cache, and no decode state on a fixed P worker."""

    @staticmethod
    def default_execution_slots(local_experts):
        return 2 * local_experts

    def validate_topology(self, owner):
        parallel = get_parallel()
        if parallel.pp_size == 1 and parallel.attn_cp_size == 1:
            return
        # Keep the new path narrow: native stage groups, with CP introduced
        # only after PP4/CP1 has been validated. Existing PP1 stays unchanged.
        if (
            owner.model_id != "GLM-5.2"
            or parallel.pp_size != 4
            or parallel.tp_size != 4
            or parallel.moe_ep_size != 4
            or parallel.dp_size != 1
            or parallel.attn_dp_size != 1
            or parallel.attn_cp_size not in (1, 4)
        ):
            raise ValueError(
                "Pipelined prefill-only offload requires GLM-5.2, "
                "PP4/TP4/EP4, DP1/attention DP1 and attention CP1 or CP4"
            )
        if parallel.disable_attn_tp_gather:
            raise ValueError(
                "Pipelined prefill-only offload requires scheduler phase "
                "synchronization; remove --disable-attn-tp-gather"
            )
        if parallel.attn_cp_size == 4 and (
            not parallel.enable_prefill_cp
            or parallel.cp_strategy != "zigzag"
            or parallel.attn_tp_size != 1
        ):
            raise ValueError("Prefill-only CP4 requires zigzag CP and attention TP1")
        if (
            parallel.enable_cp_tp_group_sharing
            or parallel.enable_dsa_cache_layer_split
            or parallel.enable_kv_cache_sharding
            or parallel.cp_kv_layout is not None
            or get_spec().speculative_algorithm
        ):
            raise ValueError(
                "Pipelined prefill-only offload requires MTP, CP-TP group "
                "sharing and KV sharding/layer splitting to be disabled"
            )

    def validate_quantization(self, owner, quant_spec):
        if get_parallel().pp_size > 1 and quant_spec.kind != "w8a8":
            raise ValueError("Pipelined prefill-only offload requires ModelSlim W8A8")

    def configure(self, owner, path):
        owner.hot_slots = 0
        parallel = get_parallel()
        if parallel.attn_dp_size <= 0 or parallel.tp_size % parallel.attn_dp_size:
            raise ValueError("Attention DP must divide TP for prefill-only offload")
        if parallel.attn_dp_size > 1 and envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get():
            raise ValueError(
                "Prefill-only offload requires scheduler DP synchronization; "
                "unset SGLANG_SCHEDULER_SKIP_ALL_GATHER"
            )
        graphs = get_exec().graph.cuda_graph_config
        if any(
            phase.backend != "disabled" for phase in (graphs.prefill, graphs.decode)
        ):
            raise ValueError(
                "Prefill-only expert offload requires --disable-cuda-graph"
            )

    def post_init(self, owner):
        from .prefill import AscendPrefillWeights

        owner.direct_hot = False
        owner.execution_keys = None
        owner.decode_active_rows = None
        owner.hot = owner.state = None
        owner.execution = owner.format.allocate_bank(
            owner.execution_slots, ResourceKind.EXECUTION_WEIGHTS, "ascend.execution"
        )
        owner.error = torch.zeros((), dtype=torch.int32, device=owner.device)
        owner.weight_copy_stream = torch.npu.Stream(device=owner.device)
        owner.context = owner._new_context()
        owner.prefill = AscendPrefillWeights(owner)
        owner.ready = True
        logger.info(
            "Ascend prefill-only offload: layers=%d local=%d hot=0 "
            "execution=%.3f GiB host=%.3f GiB",
            len(owner.catalog.offloaded_layers),
            owner.local_experts,
            owner.format.expert_nbytes * owner.execution_slots / 1024**3,
            owner.host_pool.used_bytes / 1024**3,
        )

    @contextmanager
    def forward_scope(self, owner, batch):
        if not owner.ready:
            yield
            return
        if batch is None:
            raise ValueError("Prefill-only offload requires the original forward batch")
        # The global phase is known before ModelRunner installs RuntimeContext.
        # An idle attention rank still receives other ranks' expert work.
        if not batch.is_extend_in_batch:
            # The scheduler skips globally idle rounds. Silently yielding here
            # would let an unexpected idle forward enter the MC2/decode path.
            raise ValueError("Prefill-only offload requires a globally extending batch")
        owner.context = owner._new_context()
        try:
            owner.prefill.pipeline.begin_forward(owner.context)
            yield
            owner.prefill.pipeline.finish_forward(owner.context)
            owner._check_error()
        finally:
            # Includes a partially submitted begin_forward. Never release Host
            # owners or reset bank tickets until all queued readers have drained.
            owner.context.compute_stream.wait_stream(owner.weight_copy_stream)
            owner.context.record_compute_completion().completion.synchronize()
            owner.context.resources.clear()
            owner.prefill.pipeline.reset()

    def prepare_decode_weights(self, owner, layer_id, runner_input):
        raise RuntimeError("Prefill-only offload has no decode execution path")
