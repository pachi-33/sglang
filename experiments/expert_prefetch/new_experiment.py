#!/usr/bin/env python3
"""Create an indexed expert-prefetch experiment record."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

EXPERIMENT_ID_RE = re.compile(r"^EXP-(\d{4,})$")
TYPE_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
TIMEZONE = ZoneInfo("Asia/Singapore")
INDEX_FIELDS = (
    "experiment_id",
    "type",
    "started_at",
    "ended_at",
    "status",
    "title",
    "device",
    "git_commit",
    "path",
)


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "experiment"


def _read_rows(index_path: Path) -> list[dict[str, str]]:
    if not index_path.exists():
        return []
    with index_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(INDEX_FIELDS):
            raise ValueError(
                f"unexpected INDEX.csv columns: {reader.fieldnames}; "
                f"expected {list(INDEX_FIELDS)}"
            )
        return list(reader)


def _next_id(rows: list[dict[str, str]]) -> str:
    numbers = []
    for row in rows:
        match = EXPERIMENT_ID_RE.fullmatch(row["experiment_id"])
        if not match:
            raise ValueError(
                f"invalid experiment_id in INDEX.csv: {row['experiment_id']}"
            )
        numbers.append(int(match.group(1)))
    return f"EXP-{max(numbers, default=0) + 1:04d}"


def _write_metadata(path: Path, metadata: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _readme_template(metadata: dict[str, object]) -> str:
    return f"""# {metadata['experiment_id']}: {metadata['title']}

## 实验身份

- 类型：`{metadata['type']}`
- 开始时间：`{metadata['started_at']}`
- 结束时间：待填写
- 状态：`planned`
- 设备：{metadata['device'] or '待填写'}
- Git：`{metadata['git_branch']}@{metadata['git_commit']}`，dirty={str(metadata['git_dirty']).lower()}

## 问题与假设

<!-- 要回答的问题，以及可被数据证伪的假设。 -->

## 对照与变量

- 对照组：
- 独立变量：
- 固定变量：模型、ExpertPack、输入 token、生成参数、cache 容量、设备等。
- 干扰因素：

## 指标与通过条件

<!-- 正确性、prefetch useful/late/wasted、I/O、H2D、TTFT、ITL、显存等。 -->

## 操作步骤

1. 

## 结果

<!-- 链接 results/ 下的汇总文件，并写出足以判断假设的关键数值。 -->

## 结论与后续

<!-- 接受/拒绝假设、局限，以及下一实验编号。 -->

## 资源清理

- [ ] 临时 API/CLI 进程已关闭
- [ ] 测试 GPU 无遗留计算进程
- [ ] 大体积日志和产物位置已记录
"""


def create_experiment(
    ledger_root: Path, experiment_type: str, title: str, device: str
) -> Path:
    if not TYPE_RE.fullmatch(experiment_type):
        raise ValueError("--type must use lowercase kebab-case")

    ledger_root = ledger_root.resolve()
    repo_root = Path(_git(ledger_root, "rev-parse", "--show-toplevel"))
    index_path = ledger_root / "INDEX.csv"
    lock_path = ledger_root / ".index.lock"
    now = datetime.now(TIMEZONE)
    git_status = _git(repo_root, "status", "--porcelain")
    git_branch = _git(repo_root, "branch", "--show-current")
    git_commit = _git(repo_root, "rev-parse", "HEAD")

    ledger_root.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        rows = _read_rows(index_path)
        experiment_id = _next_id(rows)
        directory_name = (
            f"{experiment_id}__{now.strftime('%Y%m%dT%H%M%S%z')}__{_slugify(title)}"
        )
        relative_path = Path("records") / experiment_type / directory_name
        record_path = ledger_root / relative_path
        record_path.mkdir(parents=True, exist_ok=False)
        for child in ("results", "logs", "artifacts"):
            child_path = record_path / child
            child_path.mkdir()
            (child_path / ".gitkeep").touch()

        metadata: dict[str, object] = {
            "format": "SGLANG-EXPERT-PREFETCH-EXPERIMENT-v1",
            "experiment_id": experiment_id,
            "type": experiment_type,
            "title": title,
            "status": "planned",
            "started_at": now.isoformat(timespec="seconds"),
            "ended_at": None,
            "timezone": "Asia/Singapore",
            "device": device or None,
            "hostname": socket.gethostname(),
            "git_branch": git_branch,
            "git_commit": git_commit,
            "git_dirty": bool(git_status),
        }
        _write_metadata(record_path / "metadata.json", metadata)
        (record_path / "README.md").write_text(
            _readme_template(metadata), encoding="utf-8"
        )
        commands_path = record_path / "commands.sh"
        commands_path.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n\n# Record exact commands below.\n",
            encoding="utf-8",
        )
        commands_path.chmod(0o755)

        new_row = {
            "experiment_id": experiment_id,
            "type": experiment_type,
            "started_at": metadata["started_at"],
            "ended_at": "",
            "status": "planned",
            "title": title,
            "device": device,
            "git_commit": metadata["git_commit"],
            "path": relative_path.as_posix(),
        }
        write_header = not index_path.exists()
        with index_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, fieldnames=INDEX_FIELDS, lineterminator="\n"
            )
            if write_header:
                writer.writeheader()
            writer.writerow(new_row)
            file.flush()
            os.fsync(file.fileno())

    return record_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Allocate an ID and create an expert-prefetch experiment record."
    )
    parser.add_argument("--type", required=True, help="lowercase kebab-case type")
    parser.add_argument("--title", required=True)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--ledger-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    try:
        record_path = create_experiment(
            args.ledger_root, args.type, args.title, args.device
        )
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(record_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
