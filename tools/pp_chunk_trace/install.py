#!/usr/bin/env python3
"""Install/uninstall the opt-in bootstrap without modifying SGLang sources."""

import argparse
import json
import os
from pathlib import Path
import sysconfig
import tempfile


PTH_NAME = "zz_pp_chunk_trace.pth"
MARKER = "# Managed by pp_chunk_trace/install.py; format=1"
BOOTSTRAP = (
    "import os; os.environ.get('PP_CHUNK_TRACE_ENABLE') == '1' and "
    "__import__('pp_chunk_trace.bootstrap', fromlist=['activate']).activate()"
)


def _payload(source):
    source = str(Path(source).resolve())
    if "\n" in source or "\r" in source:
        raise ValueError("Package path must not contain newlines")
    return MARKER + "\n" + source + "\n" + BOOTSTRAP + "\n"


def _read_owned(path):
    if path.is_symlink():
        raise ValueError(f"Refusing to modify a symlink: {path}")
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if (
        len(lines) != 3
        or lines[0] != MARKER
        or lines[2] != BOOTSTRAP
        or not Path(lines[1]).is_absolute()
    ):
        raise ValueError(f"Refusing to modify an unrecognized file: {path}")
    return Path(lines[1])


def status(target):
    path = Path(target) / PTH_NAME
    source = _read_owned(path)
    return {
        "installed": source is not None,
        "pth": str(path),
        "source": str(source) if source is not None else None,
        "source_exists": (source / "pp_chunk_trace" / "bootstrap.py").is_file()
        if source is not None else False,
        "enable": "PP_CHUNK_TRACE_ENABLE=1",
        "restart_required": True,
    }


def install(target, source=None):
    target = Path(target).resolve()
    source = Path(source or Path(__file__).parent).resolve()
    if not (source / "pp_chunk_trace" / "bootstrap.py").is_file():
        raise ValueError(f"Missing pp_chunk_trace/bootstrap.py under {source}")
    path = target / PTH_NAME
    previous = _read_owned(path)
    if previous is not None:
        if previous != source:
            raise ValueError(
                f"Already installed from {previous}; uninstall that registration "
                "before installing a different source directory"
            )
        return status(target)
    target.mkdir(parents=True, exist_ok=True)
    # Publish a complete file atomically. link() refuses to replace a file that
    # appeared after the ownership check (including a concurrent install).
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target, prefix=".pp_chunk_trace-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(_payload(source))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
        os.link(str(temporary), str(path))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return status(target)


def uninstall(target):
    target = Path(target).resolve()
    path = target / PTH_NAME
    if _read_owned(path) is not None:
        path.unlink()
    return status(target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("install", "uninstall", "status"))
    parser.add_argument(
        "--target", type=Path, default=Path(sysconfig.get_path("purelib")),
        help="Site-packages directory used by the server's Python interpreter",
    )
    args = parser.parse_args(argv)
    try:
        result = {"install": install, "uninstall": uninstall, "status": status}[
            args.command
        ](args.target)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"PP chunk trace installer: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
