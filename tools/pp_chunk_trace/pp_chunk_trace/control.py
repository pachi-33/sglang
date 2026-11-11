"""Request local trace finalization after the measured requests have completed."""
import argparse
import json
import time

from .recorder import run_directory


def control(command, directory, run_id, timeout=15.0):
    root = run_directory(directory, run_id)
    files = sorted(root.glob("*.jsonl"))
    if not files:
        raise ValueError(f"No rank traces in {root}; check enable/install status")
    if command == "start" and (root / ".finish").exists():
        raise ValueError("Capture is finished; restart workers with a new run ID")
    (root / ("." + command)).touch(exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        pending = []
        errors = []
        for path in files:
            records = []
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    records.append(json.loads(line))
                except ValueError:
                    pass
            event = "session_end" if command == "finish" else "capture_start"
            ended = any(r.get("type") == event for r in records)
            if not ended:
                pending.append(path.name)
            if any(r.get("type") in ("error", "dropped") or
                   (r.get("type") == "forward" and r.get("status") != "ok") or
                   (r.get("type") == "session_end" and not r.get("complete", False))
                   for r in records):
                errors.append(path.name)
        if not pending or time.monotonic() >= deadline:
            return dict(directory=str(root), workers=len(files), pending=pending, errors=errors)
        time.sleep(0.1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["start", "finish"])
    parser.add_argument("--directory", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout", type=float, default=15)
    args = parser.parse_args(argv)
    if not 0 <= args.timeout <= 60:
        parser.error("timeout must be in [0,60] seconds")
    try:
        result = control(args.command, args.directory, args.run_id, args.timeout)
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["pending"] or result["errors"]:
        parser.exit(1, "Capture is incomplete; preserve logs and inspect warnings.\n")


if __name__ == "__main__":
    main()
