"""Low-overhead runtime recorder for decode-time MoE traces.

Router taps write into one set of flat, preallocated device buffers.  Decode
CUDA graphs capture those writes.  At the end of a real decode forward the
valid rows are cloned, which gives the scheduler an ownership-safe payload for
its normal asynchronous D2H path without synchronizing the forward stream.
"""

from __future__ import annotations

import contextlib
import logging
from collections import defaultdict
from typing import Iterator, Optional, Sequence

import torch

from sglang.srt.moe_trace.batch_writer import MoeTraceBatchWriter
from sglang.srt.moe_trace.codec import quantize_and_pack_int4
from sglang.srt.moe_trace.config import MoeTraceConfig
from sglang.srt.moe_trace.types import (
    MoeTraceBatchOutput,
    MoeTraceForwardContext,
    MoeTraceSite,
    MoeTraceSiteLayout,
    TraceRowLayout,
)
from sglang.srt.moe_trace.writer import MoeTraceWriter
from sglang.srt.runtime_context import get_parallel, get_resources

logger = logging.getLogger(__name__)


def build_site_layouts(
    sites: Sequence[MoeTraceSite], group_size: int
) -> tuple[MoeTraceSiteLayout, ...]:
    q_offset = scale_offset = route_offset = 0
    layouts: list[MoeTraceSiteLayout] = []
    for site in sites:
        q_width = (site.hidden_size + 1) // 2
        scale_width = (site.hidden_size + group_size - 1) // group_size
        layouts.append(
            MoeTraceSiteLayout(
                site=site,
                activation_q_offset=q_offset,
                activation_q_width=q_width,
                activation_scale_offset=scale_offset,
                activation_scale_width=scale_width,
                route_offset=route_offset,
            )
        )
        q_offset += q_width
        scale_offset += scale_width
        route_offset += site.top_k
    return tuple(layouts)


class MoeTraceRecorder:
    """Process-local device recorder and asynchronous file writer."""

    def __init__(
        self,
        config: MoeTraceConfig,
        sites: Sequence[MoeTraceSite],
        *,
        max_rows: int,
        device: torch.device | str,
    ) -> None:
        config.validate()
        if max_rows <= 0:
            raise ValueError("MoE trace max_rows must be positive")
        if not sites:
            raise ValueError("MoE tracing was enabled but no routed MoE sites exist")
        self.config = config
        self.layouts = build_site_layouts(sites, config.activation_group_size)
        self._layouts_by_id = {layout.site.site_id: layout for layout in self.layouts}
        if set(self._layouts_by_id) != set(range(len(self.layouts))):
            raise ValueError("MoE trace site IDs must be contiguous from zero")
        self.max_rows = max_rows
        self.device = torch.device(device)
        self.capture_enabled = False
        self._trace_counts: dict[str, int] = defaultdict(int)
        self._finalized: set[str] = set()
        self._pending_finalization: set[str] = set()
        self._next_forward_id = 0

        num_sites = len(self.layouts)
        self.site_valid = torch.zeros(
            (max_rows, num_sites), dtype=torch.bool, device=self.device
        )
        self.activation_q = self.activation_scales = None
        self.expert_ids = self.expert_weights = None
        if config.router_inputs:
            q_width = sum(layout.activation_q_width for layout in self.layouts)
            scale_width = sum(layout.activation_scale_width for layout in self.layouts)
            self.activation_q = torch.empty(
                (max_rows, q_width), dtype=torch.uint8, device=self.device
            )
            self.activation_scales = torch.empty(
                (max_rows, scale_width), dtype=torch.float16, device=self.device
            )
        if config.expert_routes:
            route_width = sum(layout.site.top_k for layout in self.layouts)
            self.expert_ids = torch.empty(
                (max_rows, route_width), dtype=torch.int32, device=self.device
            )
            if not config.ids_only:
                self.expert_weights = torch.empty(
                    (max_rows, route_width), dtype=torch.float32, device=self.device
                )

        writer_class = MoeTraceBatchWriter if config.ids_only else MoeTraceWriter
        self.writer = writer_class(
            config.output_dir,
            self.layouts,
            rank=get_parallel().tp_rank,
            queue_depth=config.queue_depth,
            overflow=config.overflow_policy,
            **(
                {"max_rows_per_request": config.max_decode_tokens}
                if config.ids_only
                else {}
            ),
            quantization=(
                {
                    "scheme": "symmetric-groupwise-int4",
                    "group_size": config.activation_group_size,
                    "scale_dtype": "float16",
                    "encoding": "two-complement-nibbles-low-first",
                    "quant_range": [-7, 7],
                }
                if config.router_inputs
                else {}
            ),
        )
        logger.info(
            "Enabled decode MoE tracing: sites=%d max_rows=%d routes=%s "
            "router_inputs=%s output=%s",
            len(self.layouts),
            max_rows,
            config.expert_routes,
            config.router_inputs,
            config.output_dir,
        )

    @contextlib.contextmanager
    def capture_scope(self, enabled: bool = True) -> Iterator[None]:
        previous = self.capture_enabled
        self.capture_enabled = enabled
        try:
            yield
        finally:
            self.capture_enabled = previous

    def begin_forward(self, enabled: bool) -> None:
        if enabled:
            # The clear is deliberately outside a CUDA graph. Captured router
            # writes set every supported site's column during replay.
            self.site_valid.zero_()

    def _layout_for(self, module: torch.nn.Module) -> MoeTraceSiteLayout | None:
        site_id = getattr(module, "_moe_trace_site_id", None)
        if site_id is None:
            return None
        return self._layouts_by_id.get(int(site_id))

    def capture_router_input(
        self, module: torch.nn.Module, hidden_states: torch.Tensor
    ) -> None:
        if not self.capture_enabled or not self.config.router_inputs:
            return
        layout = self._layout_for(module)
        if layout is None:
            return
        if hidden_states.ndim != 2 or hidden_states.shape[1] != layout.site.hidden_size:
            raise RuntimeError(
                f"MoE trace site {layout.site.site_id} expected router input "
                f"[rows, {layout.site.hidden_size}], got {tuple(hidden_states.shape)}"
            )
        rows = hidden_states.shape[0]
        if rows > self.max_rows:
            raise RuntimeError(
                f"MoE trace buffer has {self.max_rows} rows but site "
                f"{layout.site.site_id} produced {rows}"
            )
        q = self.activation_q[
            :rows,
            layout.activation_q_offset : layout.activation_q_offset
            + layout.activation_q_width,
        ]
        scales = self.activation_scales[
            :rows,
            layout.activation_scale_offset : layout.activation_scale_offset
            + layout.activation_scale_width,
        ]
        quantize_and_pack_int4(
            hidden_states,
            self.config.activation_group_size,
            out_q=q,
            out_scales=scales,
        )
        self.site_valid[:rows, layout.site.site_id] = True

    def capture_route(
        self,
        module: torch.nn.Module,
        logical_topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        if not self.capture_enabled or not self.config.expert_routes:
            return
        layout = self._layout_for(module)
        if layout is None:
            return
        expected = (logical_topk_ids.shape[0], layout.site.top_k)
        if logical_topk_ids.ndim != 2 or tuple(logical_topk_ids.shape) != expected:
            raise RuntimeError(
                f"MoE trace site {layout.site.site_id} expected route shape "
                f"[rows, {layout.site.top_k}], got {tuple(logical_topk_ids.shape)}"
            )
        if tuple(topk_weights.shape) != expected:
            raise RuntimeError(
                f"MoE trace site {layout.site.site_id} route weights do not "
                "match expert IDs"
            )
        rows = expected[0]
        if rows > self.max_rows:
            raise RuntimeError(
                f"MoE trace buffer has {self.max_rows} rows but site "
                f"{layout.site.site_id} produced {rows}"
            )
        route = slice(layout.route_offset, layout.route_offset + layout.site.top_k)
        self.expert_ids[:rows, route].copy_(logical_topk_ids)
        if self.expert_weights is not None:
            self.expert_weights[:rows, route].copy_(topk_weights)
        self.site_valid[:rows, layout.site.site_id] = True

    def _selected_rows(self, request_ids: Sequence[str]) -> list[int]:
        selected: list[int] = []
        limit = self.config.max_decode_tokens
        for row, request_id in enumerate(request_ids):
            count = self._trace_counts[request_id]
            if limit == 0 or count < limit:
                self._trace_counts[request_id] = count + 1
                selected.append(row)
        return selected

    @staticmethod
    def _clone_rows(tensor: torch.Tensor, rows: int, indices: torch.Tensor | None):
        view = tensor[:rows]
        return view.clone() if indices is None else view.index_select(0, indices)

    def prepare_forward(self, forward_batch) -> MoeTraceForwardContext:
        """Freeze row ownership and inputs before padding/rebinding in forward."""
        request_ids = tuple(forward_batch.rids or ())
        if not request_ids:
            raise ValueError("target trace requires request IDs for every active batch")
        if forward_batch.forward_mode.is_decode():
            phase, width = "decode", 1
        elif forward_batch.forward_mode.is_target_verify():
            phase = "target_verify"
            width = int(getattr(forward_batch.spec_info, "num_tokens_per_req", 0))
            if width <= 0:
                raise ValueError("target verify requires an explicit trace row width")
        else:
            raise ValueError("MoE IDs-only tracing requires decode or target verify")
        layout = TraceRowLayout.from_counts(
            self._next_forward_id, phase, request_ids, [width] * len(request_ids)
        )
        self._next_forward_id += 1
        return self.prepare_rows(
            layout, forward_batch.input_ids, forward_batch.positions
        )

    def prepare_rows(
        self, layout, input_token_ids, positions
    ) -> MoeTraceForwardContext:
        """Also accepts explicit nonuniform layouts from future verify adapters."""
        maximum = max(layout.capture_indices, default=-1)
        if maximum >= self.max_rows:
            raise ValueError("trace row layout exceeds the device buffer capacity")
        if any(
            tensor.ndim != 1 or tensor.shape[0] <= maximum
            for tensor in (input_token_ids, positions)
        ):
            raise ValueError("trace row layout exceeds token/position metadata")
        if layout.capture_indices == tuple(range(len(layout.capture_indices))):
            rows = len(layout.capture_indices)
            return MoeTraceForwardContext(
                layout, input_token_ids[:rows].clone(), positions[:rows].clone()
            )
        indices = torch.tensor(
            layout.capture_indices, dtype=torch.long, device=self.device
        )
        return MoeTraceForwardContext(
            layout,
            input_token_ids.index_select(0, indices),
            positions.index_select(0, indices),
        )

    def _end_ids_forward(self, context) -> Optional[MoeTraceBatchOutput]:
        layout = context.layout
        request_ids = layout.expanded_request_ids()
        selected = self._selected_rows(request_ids)
        if not selected:
            return None
        if len(selected) == len(request_ids) and layout.capture_indices == tuple(
            range(len(request_ids))
        ):
            rows = len(request_ids)
            return MoeTraceBatchOutput(
                request_ids=request_ids,
                input_token_ids=context.input_token_ids,
                positions=context.positions,
                site_valid=self.site_valid[:rows].clone(),
                expert_ids=self.expert_ids[:rows].clone(),
                row_layout=layout,
            )
        layout = layout.select_rows(selected)
        capture_indices = torch.tensor(
            layout.capture_indices, dtype=torch.long, device=self.device
        )
        metadata_indices = torch.tensor(selected, dtype=torch.long, device=self.device)
        return MoeTraceBatchOutput(
            request_ids=layout.expanded_request_ids(),
            input_token_ids=context.input_token_ids.index_select(0, metadata_indices),
            positions=context.positions.index_select(0, metadata_indices),
            site_valid=self.site_valid.index_select(0, capture_indices),
            expert_ids=self.expert_ids.index_select(0, capture_indices),
            row_layout=layout,
        )

    def end_forward(self, forward_batch, context=None) -> Optional[MoeTraceBatchOutput]:
        if self.config.ids_only:
            return self._end_ids_forward(context or self.prepare_forward(forward_batch))
        if not forward_batch.forward_mode.is_decode():
            return None
        request_ids = list(forward_batch.rids or ())
        rows = forward_batch.batch_size
        if len(request_ids) != rows:
            raise RuntimeError(
                "MoE trace decode metadata requires one request ID per batch row"
            )
        selected = self._selected_rows(request_ids)
        if not selected:
            return None
        all_rows = len(selected) == rows
        indices = None
        if not all_rows:
            indices = torch.tensor(selected, dtype=torch.long, device=self.device)
        request_ids = [request_ids[i] for i in selected]

        def take(tensor: torch.Tensor | None):
            return None if tensor is None else self._clone_rows(tensor, rows, indices)

        return MoeTraceBatchOutput(
            request_ids=request_ids,
            input_token_ids=self._clone_rows(forward_batch.input_ids, rows, indices),
            positions=self._clone_rows(forward_batch.positions, rows, indices),
            site_valid=self._clone_rows(self.site_valid, rows, indices),
            activation_q=take(self.activation_q),
            activation_scales=take(self.activation_scales),
            expert_ids=take(self.expert_ids),
            expert_weights=take(self.expert_weights),
        )

    @staticmethod
    def _select_payload_rows(
        payload: MoeTraceBatchOutput, selected: list[int]
    ) -> MoeTraceBatchOutput:
        device = payload.input_token_ids.device
        indices = torch.tensor(selected, dtype=torch.long, device=device)

        def take(tensor: torch.Tensor | None):
            return None if tensor is None else tensor.index_select(0, indices)

        filtered = MoeTraceBatchOutput(
            request_ids=[payload.request_ids[index] for index in selected],
            input_token_ids=take(payload.input_token_ids),
            positions=take(payload.positions),
            site_valid=take(payload.site_valid),
            activation_q=take(payload.activation_q),
            activation_scales=take(payload.activation_scales),
            expert_ids=take(payload.expert_ids),
            expert_weights=take(payload.expert_weights),
        )
        payload.release()
        return filtered

    def submit(self, payload: MoeTraceBatchOutput) -> bool:
        """Submit a trace batch and retire one overlap-scheduler lookahead.

        At request completion the overlap scheduler may already have launched
        one more decode.  Completion is therefore deferred until the next
        result reaches this method; matching rows are overshoot and are not
        persisted.
        """
        if self.config.ids_only:
            # Execution traces retain already-launched overlap forwards, even
            # after logical request completion. No per-request overshoot filter.
            return self.writer.submit(payload)
        pending = set(self._pending_finalization)
        accepted = False
        if pending:
            selected = [
                row
                for row, request_id in enumerate(payload.request_ids)
                if request_id not in pending
            ]
            if selected:
                if len(selected) != len(payload.request_ids):
                    payload = self._select_payload_rows(payload, selected)
                accepted = self.writer.submit(payload)
            else:
                payload.release()
            self.flush_pending_finalizations()
            return accepted
        return self.writer.submit(payload)

    def finalize_request(
        self,
        request_id: str,
        *,
        status: str = "complete",
        error: str | None = None,
        defer: bool = False,
    ) -> None:
        if (
            request_id in self._finalized
            or request_id in self._pending_finalization
            or self._trace_counts.get(request_id, 0) == 0
        ):
            return
        defer = defer and not self.config.ids_only
        if defer:
            self._pending_finalization.add(request_id)
            return
        self._finalized.add(request_id)
        self.writer.finalize_request(request_id, status=status, error=error)

    def flush_pending_finalizations(self) -> None:
        pending, self._pending_finalization = self._pending_finalization, set()
        for request_id in pending:
            if request_id not in self._finalized:
                self._finalized.add(request_id)
                self.writer.finalize_request(request_id, status="complete")

    def close(self) -> None:
        # Anything still open at shutdown remains a useful, explicitly
        # truncated trace rather than an ambiguous "open" manifest.
        self.flush_pending_finalizations()
        for request_id in tuple(self._trace_counts):
            if request_id not in self._finalized:
                self.finalize_request(request_id, status="truncated")
        self.writer.close()


def get_global_moe_trace_recorder() -> Optional[MoeTraceRecorder]:
    return get_resources().moe_trace_recorder


def set_global_moe_trace_recorder(recorder: Optional[MoeTraceRecorder]) -> None:
    get_resources().moe_trace_recorder = recorder


def destroy_global_moe_trace_recorder() -> None:
    recorder = get_global_moe_trace_recorder()
    if recorder is not None:
        recorder.close()
    set_global_moe_trace_recorder(None)


def capture_router_input(module: torch.nn.Module, hidden_states: torch.Tensor) -> None:
    recorder = get_global_moe_trace_recorder()
    if recorder is not None:
        recorder.capture_router_input(module, hidden_states)


def capture_route(
    module: torch.nn.Module,
    logical_topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> None:
    recorder = get_global_moe_trace_recorder()
    if recorder is not None:
        recorder.capture_route(module, logical_topk_ids, topk_weights)


__all__ = [
    "MoeTraceRecorder",
    "build_site_layouts",
    "capture_route",
    "capture_router_input",
    "destroy_global_moe_trace_recorder",
    "get_global_moe_trace_recorder",
    "set_global_moe_trace_recorder",
]
