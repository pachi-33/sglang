#!/usr/bin/env bash
# Safely reclaim clean page cache and report the before/after state of selected
# NUMA nodes. The Linux drop_caches interface is global, so nodes 0-7 are
# reported by default even though only the requested nodes are shown.
set -Eeuo pipefail

SCRIPT_NAME=$(basename -- "$0")
SETTLE_SECONDS=${SETTLE_SECONDS:-8}
DIRTY_ABORT_KIB=${DIRTY_ABORT_KIB:-262144}
LOCK_FILE=${LOCK_FILE:-/tmp/safe_release_numa_mem.lock}

APPLY=0
FORCE=0
NODES=()

usage() {
  cat <<EOF
Usage:
  $SCRIPT_NAME <node> [node ...] [--apply] [--force]

Examples:
  $SCRIPT_NAME 0 1 2 3
      Dry run: show current memory and safety checks. Makes no changes.

  $SCRIPT_NAME 0 1 2 3 --apply
      Run sync, then reclaim clean page cache with drop_caches=1.

Options:
  --apply   Actually perform the cleanup. Without it, this is a dry run.
  --force   Bypass the active SGLang/Profiler and dirty-page safety checks.

Environment:
  SETTLE_SECONDS     Seconds to wait before the after snapshot (default: 8).
  DIRTY_ABORT_KIB    Abort threshold after sync (default: 262144 KiB).

Safety:
  - drop_caches=1 evicts clean file pages only; it does not clear slab.
  - The kernel interface is global and cannot be restricted to one NUMA node.
  - Do not run this while a model is actively loading or serving.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

for arg in "$@"; do
  case "$arg" in
    --apply)
      APPLY=1
      ;;
    --force)
      FORCE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      [[ "$arg" =~ ^[0-9]+$ ]] ||
        die "invalid NUMA node '$arg'; expected a non-negative integer"
      NODES+=("$arg")
      ;;
  esac
done

if ((${#NODES[@]} == 0)); then
  usage
  exit 2
fi

[[ "$SETTLE_SECONDS" =~ ^[0-9]+$ ]] ||
  die "SETTLE_SECONDS must be a non-negative integer"
[[ "$DIRTY_ABORT_KIB" =~ ^[0-9]+$ ]] ||
  die "DIRTY_ABORT_KIB must be a non-negative integer"

declare -A SEEN_NODE=()
UNIQUE_NODES=()
for node in "${NODES[@]}"; do
  meminfo="/sys/devices/system/node/node${node}/meminfo"
  [[ -r "$meminfo" ]] || die "NUMA node${node} is not available ($meminfo)"
  if [[ -z "${SEEN_NODE[$node]+x}" ]]; then
    SEEN_NODE["$node"]=1
    UNIQUE_NODES+=("$node")
  fi
done
NODES=("${UNIQUE_NODES[@]}")

node_field() {
  local node=$1
  local field=$2
  awk -v node="$node" -v field="$field" '
    $1 == "Node" && $2 == node && $3 == field ":" {
      print $4
      found = 1
      exit
    }
    END {
      if (!found)
        exit 1
    }
  ' "/sys/devices/system/node/node${node}/meminfo"
}

global_field() {
  local field=$1
  awk -v field="$field" '
    $1 == field ":" {
      print $2
      found = 1
      exit
    }
    END {
      if (!found)
        exit 1
    }
  ' /proc/meminfo
}

kib_to_gib() {
  awk -v kib="$1" 'BEGIN { printf "%.2f", kib / 1048576 }'
}

kib_to_mib() {
  awk -v kib="$1" 'BEGIN { printf "%.2f", kib / 1024 }'
}

free_percent() {
  awk -v free="$1" -v total="$2" '
    BEGIN {
      if (total > 0)
        printf "%.2f", free * 100 / total
      else
        printf "0.00"
    }
  '
}

collect_active_pids() {
  ACTIVE_PIDS=()
  local pid state ppid
  local -A excluded=()

  pid=$$
  while [[ "$pid" =~ ^[0-9]+$ ]] && ((pid > 1)); do
    excluded["$pid"]=1
    [[ -r "/proc/$pid/stat" ]] || break
    ppid=$(awk '{print $4}' "/proc/$pid/stat" 2>/dev/null || true)
    [[ "$ppid" =~ ^[0-9]+$ ]] || break
    pid=$ppid
  done

  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    [[ -z "${excluded[$pid]+x}" ]] || continue
    [[ -r "/proc/$pid/stat" ]] || continue
    state=$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || true)
    [[ "$state" != "Z" ]] || continue
    ACTIVE_PIDS+=("$pid")
  done < <(
    pgrep -f 'python([0-9.]*)? .*([-]m[[:space:]]+sglang\.launch_server|[p]rofiler\.py)' ||
      true
  )
}

show_active_processes() {
  collect_active_pids
  if ((${#ACTIVE_PIDS[@]} == 0)); then
    echo "Active SGLang/Profiler processes: none"
    return
  fi

  echo "Active SGLang/Profiler processes:"
  ps -o pid=,stat=,etime=,args= -p "$(IFS=,; echo "${ACTIVE_PIDS[*]}")" ||
    true
}

print_snapshot() {
  local label=$1
  local node total free file_pages dirty

  printf '\n[%s]\n' "$label"
  printf '%-6s %14s %14s %14s %10s %14s\n' \
    "NUMA" "Total(GiB)" "MemFree(GiB)" "FilePages(GiB)" "Free(%)" "Dirty(MiB)"
  for node in "${NODES[@]}"; do
    total=$(node_field "$node" MemTotal)
    free=$(node_field "$node" MemFree)
    file_pages=$(node_field "$node" FilePages)
    dirty=$(node_field "$node" Dirty)
    printf '%-6s %14s %14s %14s %10s %14s\n' \
      "node$node" \
      "$(kib_to_gib "$total")" \
      "$(kib_to_gib "$free")" \
      "$(kib_to_gib "$file_pages")" \
      "$(free_percent "$free" "$total")" \
      "$(kib_to_mib "$dirty")"
  done
}

print_global_snapshot() {
  printf 'Global: MemFree=%s GiB, MemAvailable=%s GiB, Cached=%s GiB, Dirty=%s MiB, Writeback=%s MiB\n' \
    "$(kib_to_gib "$(global_field MemFree)")" \
    "$(kib_to_gib "$(global_field MemAvailable)")" \
    "$(kib_to_gib "$(global_field Cached)")" \
    "$(kib_to_mib "$(global_field Dirty)")" \
    "$(kib_to_mib "$(global_field Writeback)")"
}

declare -A BEFORE_TOTAL=()
declare -A BEFORE_FREE=()

echo "Requested NUMA nodes: ${NODES[*]}"
echo "Mode: $([[ "$APPLY" == 1 ]] && echo APPLY || echo DRY-RUN)"
show_active_processes
print_snapshot "Before"
print_global_snapshot

for node in "${NODES[@]}"; do
  BEFORE_TOTAL["$node"]=$(node_field "$node" MemTotal)
  BEFORE_FREE["$node"]=$(node_field "$node" MemFree)
done

if [[ "$APPLY" != 1 ]]; then
  cat <<'EOF'

DRY-RUN: no memory was released.
Run the same command with --apply to perform the cleanup.
EOF
  exit 0
fi

((EUID == 0)) || die "--apply requires root privileges"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  die "another cleanup is already running (lock: $LOCK_FILE)"
fi

collect_active_pids
if ((${#ACTIVE_PIDS[@]} > 0)) && [[ "$FORCE" != 1 ]]; then
  die "active SGLang/Profiler process detected; wait for it to finish or use --force"
fi

echo
echo "Running sync..."
sync

dirty_kib=$(global_field Dirty)
if ((dirty_kib > DIRTY_ABORT_KIB)) && [[ "$FORCE" != 1 ]]; then
  die "dirty pages remain after sync (${dirty_kib} KiB); wait or use --force"
fi

echo "Reclaiming clean page cache (drop_caches=1)..."
printf '1\n' > /proc/sys/vm/drop_caches
sleep "$SETTLE_SECONDS"

print_snapshot "After"
print_global_snapshot

printf '\n[Released per node]\n'
printf '%-6s %14s\n' "NUMA" "Freed(GiB)"
total_freed_kib=0
for node in "${NODES[@]}"; do
  free_now=$(node_field "$node" MemFree)
  freed_kib=$((free_now - BEFORE_FREE["$node"]))
  total_freed_kib=$((total_freed_kib + freed_kib))
  printf '%-6s %14s\n' "node$node" "$(kib_to_gib "$freed_kib")"
done
printf '%-6s %14s\n' "TOTAL" "$(kib_to_gib "$total_freed_kib")"
echo
echo "Cleanup completed without clearing slab or anonymous process memory."
