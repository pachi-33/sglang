"""Validate a Qwen3.5 NVFP4 expert-pack artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .build import CheckpointExpertSource, encode_expert_payload
from .format import (
    EXPERT_LAYERS,
    EXPERTS_PER_LAYER,
    PACK_SIZE,
    PAYLOAD_SIZE,
    RECORD_COUNT,
    RECORD_STRIDE,
    ExpertPackManifest,
    load_manifest,
    record_index,
)


@dataclass(frozen=True)
class ValidationReport:
    manifest: str
    pack: str
    records_verified: int
    source_records_verified: int
    pack_sha256_verified: bool
    padding_verified: int


def _selected_indices(
    record_keys: Sequence[tuple[int, int]] | None,
) -> tuple[int, ...]:
    if record_keys is None:
        return tuple(range(RECORD_COUNT))
    indices = []
    seen = set()
    for layer_id, expert_id in record_keys:
        index = record_index(layer_id, expert_id)
        if index not in seen:
            seen.add(index)
            indices.append(index)
    if not indices:
        raise ValueError("record selection is empty")
    return tuple(indices)


def _validate_pack_bytes(
    manifest: ExpertPackManifest,
    pack_path: Path,
    selected: tuple[int, ...],
    *,
    verify_pack_sha256: bool,
    verify_padding: bool,
) -> tuple[int, bool, int]:
    if not pack_path.is_file():
        raise FileNotFoundError(pack_path)
    actual_size = pack_path.stat().st_size
    if actual_size != PACK_SIZE:
        raise ValueError(
            f"expert-pack file size {actual_size} does not match {PACK_SIZE}"
        )
    selected_set = set(selected)
    records_verified = 0
    padding_verified = 0
    padding_size = RECORD_STRIDE - PAYLOAD_SIZE
    zero_padding = bytes(padding_size)

    if verify_pack_sha256:
        pack_digest = hashlib.sha256()
        with pack_path.open("rb", buffering=0) as handle:
            for index, record in enumerate(manifest.records):
                payload = handle.read(PAYLOAD_SIZE)
                padding = handle.read(padding_size)
                if len(payload) != PAYLOAD_SIZE or len(padding) != padding_size:
                    raise OSError(f"short expert-pack record read at index {index}")
                pack_digest.update(payload)
                pack_digest.update(padding)
                if index in selected_set:
                    digest = hashlib.sha256(payload).hexdigest()
                    if digest != record.sha256:
                        raise ValueError(
                            f"expert-pack payload SHA-256 mismatch at "
                            f"({record.layer_id}, {record.expert_id})"
                        )
                    records_verified += 1
                if verify_padding:
                    if padding != zero_padding:
                        raise ValueError(
                            f"non-zero expert-pack padding at "
                            f"({record.layer_id}, {record.expert_id})"
                        )
                    padding_verified += 1
            if handle.read(1):
                raise ValueError("expert-pack contains trailing bytes")
        actual_digest = pack_digest.hexdigest()
        if actual_digest != manifest.pack_sha256:
            raise ValueError(
                f"expert-pack file SHA-256 {actual_digest} does not match manifest"
            )
        return records_verified, True, padding_verified

    descriptor = os.open(str(pack_path), os.O_RDONLY)
    try:
        for index in selected:
            record = manifest.records[index]
            raw = os.pread(descriptor, RECORD_STRIDE, record.offset)
            if len(raw) != RECORD_STRIDE:
                raise OSError(
                    f"short expert-pack record read at "
                    f"({record.layer_id}, {record.expert_id})"
                )
            payload, padding = raw[:PAYLOAD_SIZE], raw[PAYLOAD_SIZE:]
            digest = hashlib.sha256(payload).hexdigest()
            if digest != record.sha256:
                raise ValueError(
                    f"expert-pack payload SHA-256 mismatch at "
                    f"({record.layer_id}, {record.expert_id})"
                )
            records_verified += 1
            if verify_padding:
                if padding != zero_padding:
                    raise ValueError(
                        f"non-zero expert-pack padding at "
                        f"({record.layer_id}, {record.expert_id})"
                    )
                padding_verified += 1
    finally:
        os.close(descriptor)
    return records_verified, False, padding_verified


def _validate_source_bytes(
    manifest: ExpertPackManifest,
    pack_path: Path,
    model_dir: Path,
    selected: tuple[int, ...],
) -> int:
    verified = 0
    with CheckpointExpertSource(model_dir) as source:
        expected_source = manifest.source
        for field in ("config_sha256", "index_sha256", "shards"):
            if source.metadata.get(field) != expected_source.get(field):
                raise ValueError(
                    f"checkpoint source {field} does not match expert-pack manifest"
                )
        descriptor = os.open(str(pack_path), os.O_RDONLY)
        try:
            for index in selected:
                record = manifest.records[index]
                source_payload = encode_expert_payload(
                    source, record.layer_id, record.expert_id
                )
                packed_payload = os.pread(descriptor, PAYLOAD_SIZE, record.offset)
                if len(packed_payload) != PAYLOAD_SIZE:
                    raise OSError(
                        f"short expert-pack payload at "
                        f"({record.layer_id}, {record.expert_id})"
                    )
                if packed_payload != source_payload:
                    raise ValueError(
                        f"expert-pack payload differs from source at "
                        f"({record.layer_id}, {record.expert_id})"
                    )
                verified += 1
        finally:
            os.close(descriptor)
    return verified


def validate_expert_pack(
    manifest_path: str | Path,
    *,
    model_dir: str | Path | None = None,
    record_keys: Sequence[tuple[int, int]] | None = None,
    verify_pack_sha256: bool = True,
    verify_padding: bool = True,
) -> ValidationReport:
    """Validate structure, selected payloads, and optionally the source bytes.

    ``record_keys=None`` verifies every record.  A selected validation can skip
    the 17 GiB whole-file digest by setting ``verify_pack_sha256=False``; each
    selected payload is still authenticated against its manifest digest.
    """
    manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    pack_path = manifest.pack_path(manifest_path)
    selected = _selected_indices(record_keys)
    records, pack_verified, padding = _validate_pack_bytes(
        manifest,
        pack_path,
        selected,
        verify_pack_sha256=verify_pack_sha256,
        verify_padding=verify_padding,
    )
    source_records = 0
    if model_dir is not None:
        source_records = _validate_source_bytes(
            manifest, pack_path, Path(model_dir).resolve(), selected
        )
    return ValidationReport(
        manifest=str(manifest_path),
        pack=str(pack_path),
        records_verified=records,
        source_records_verified=source_records,
        pack_sha256_verified=pack_verified,
        padding_verified=padding,
    )


def _parse_record(value: str) -> tuple[int, int]:
    try:
        layer_text, expert_text = value.split(":", 1)
        key = (int(layer_text), int(expert_text))
        record_index(*key)
        return key
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"invalid record {value!r}; expected LAYER:EXPERT with layer "
            f"{EXPERT_LAYERS[0]}..{EXPERT_LAYERS[-1]} and expert "
            f"0..{EXPERTS_PER_LAYER - 1}"
        ) from exc


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="also compare selected pack records byte-for-byte with the checkpoint",
    )
    parser.add_argument(
        "--record",
        action="append",
        type=_parse_record,
        dest="records",
        help="validate only LAYER:EXPERT (repeatable); defaults to all records",
    )
    parser.add_argument(
        "--skip-pack-sha256",
        action="store_true",
        help="skip the full 17 GiB file digest (selected payload SHA-256 remains checked)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = validate_expert_pack(
        args.manifest,
        model_dir=args.model_dir,
        record_keys=args.records,
        verify_pack_sha256=not args.skip_pack_sha256,
    )
    print(json.dumps(asdict(report), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ValidationReport", "validate_expert_pack"]
