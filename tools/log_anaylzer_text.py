#!/usr/bin/env python3
"""
Deep Log Analysis for plain-text logs (app.log / app.log.1 / ...)

This script is robust to different log formats by using:
- Multiple timestamp regex patterns
- Token-based detection for event types (EVENT_CREATED, CLASSIFICATION, etc.)
- Heuristic extraction of ids (event_id/id, track_id/track, frame/frame_index)

Usage:
  python tools/deep_log_analysis_text.py --log-dir data/logs --day 2026-01-06
  python tools/deep_log_analysis_text.py --log-dir data/logs --from 2026-01-06T08:00:00Z --to 2026-01-06T10:00:00Z
  python tools/deep_log_analysis_text.py --log-dir data/logs --day 2026-01-06 --trace-track-id 12345
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Any

# ---------------------------
# File discovery
# ---------------------------

def discover_log_files(log_dir: Path, base_name: str = "app.log") -> List[Path]:
    files: List[Path] = []
    main = log_dir / base_name
    if main.exists():
        files.append(main)
    for p in log_dir.glob(base_name + ".*"):
        if p.is_file():
            files.append(p)
    return sorted(set(files), key=lambda p: p.name)

# ---------------------------
# Timestamp parsing (multiple formats)
# ---------------------------

# Common formats you might see:
# 2026-01-06 12:34:56,789 - INFO - ...
# 2026-01-06 12:34:56.789 INFO ...
# 2026-01-06T12:34:56.789Z ...
# 2026-01-06T12:34:56Z ...
TS_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"), "%Y-%m-%d %H:%M:%S,%f"),
    (re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3,6})"), "%Y-%m-%d %H:%M:%S.%f"),
    (re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"), "%Y-%m-%d %H:%M:%S"),
    (re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)"), "ISO_Z"),
    (re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)"), "ISO_Z"),
]

LEVEL_PAT = re.compile(r"\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b")

def parse_ts_from_line(line: str) -> Optional[datetime]:
    for pat, fmt in TS_PATTERNS:
        m = pat.search(line)
        if not m:
            continue
        s = m.group("ts")
        try:
            if fmt == "ISO_Z":
                return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
            # NOTE: text logs are often local time; assume UTC unless you tell me otherwise
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None

def parse_level_from_line(line: str) -> str:
    m = LEVEL_PAT.search(line)
    return m.group(1) if m else "UNKNOWN"

def floor_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)

# ---------------------------
# Event type detection (token-based)
# ---------------------------

TOKENS = [
    "EVENT_CREATED",
    "EVENT_COMMITTED",
    "EVENT_EXPIRED",
    "EVENT_SUPPRESSED",
    "CLASSIFICATION",
    "COUNT_UPDATE",
    "BACKPRESSURE",
]

def detect_type(line: str, level: str) -> str:
    if level == "ERROR":
        return "ERROR"
    for t in TOKENS:
        if t in line:
            return t
    return "OTHER"

# ---------------------------
# Heuristic ID extraction
# ---------------------------

# supports: id=123, event_id=123, track=123, track_id=123, frame=123, frame_index=123
ID_PATS = {
    "event_id": [
        re.compile(r"\bevent_id=(\d+)\b"),
        re.compile(r"\bid=(\d+)\b"),  # many messages use id= for event id
    ],
    "track_id": [
        re.compile(r"\btrack_id=(\d+)\b"),
        re.compile(r"\btrack=(\d+)\b"),
    ],
    "frame_index": [
        re.compile(r"\bframe_index=(\d+)\b"),
        re.compile(r"\bframe=(\d+)\b"),
        re.compile(r"\bframe_id=(\d+)\b"),
    ],
}

def first_int_match(pats: List[re.Pattern], line: str) -> Optional[int]:
    for p in pats:
        m = p.search(line)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None

def extract_ids(line: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    event_id = first_int_match(ID_PATS["event_id"], line)
    track_id = first_int_match(ID_PATS["track_id"], line)
    frame_index = first_int_match(ID_PATS["frame_index"], line)
    return event_id, track_id, frame_index

# classification heuristics
LABEL_PAT = re.compile(r"\blabel=([A-Za-z0-9_\-]+)\b", re.IGNORECASE)
UNKNOWN_PAT = re.compile(r"\bUnknown\b", re.IGNORECASE)
REJECTION_PAT = re.compile(r"\brejection_reason=([^,]+)", re.IGNORECASE)

def extract_classification(line: str) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Returns (label, rejection_reason, is_unknown)
    Works even if the log line isn't perfectly structured.
    """
    label = None
    rr = None

    m = LABEL_PAT.search(line)
    if m:
        label = m.group(1)

    m2 = REJECTION_PAT.search(line)
    if m2:
        rr = m2.group(1).strip()

    is_unknown = False
    if label and label.lower() == "unknown":
        is_unknown = True
    elif UNKNOWN_PAT.search(line):
        # fallback: line mentions Unknown
        is_unknown = True
        if label is None:
            label = "Unknown"

    return label, rr, is_unknown

# suppression reason heuristics
SUPPRESSION_REASON_PAT = re.compile(r"\breason=([A-Za-z0-9_\-]+)\b", re.IGNORECASE)
EXPIRED_STATE_PAT = re.compile(r"\bstate=([A-Za-z0-9_\-]+)\b", re.IGNORECASE)
BACKPRESSURE_DROPS_PAT = re.compile(r"\bdrops=(\d+)\b", re.IGNORECASE)
BACKPRESSURE_QUEUE_PAT = re.compile(r"\bqueue=([A-Za-z0-9_\-]+)\b", re.IGNORECASE)

def extract_kv_simple(pat: re.Pattern, line: str) -> Optional[str]:
    m = pat.search(line)
    return m.group(1) if m else None

# ---------------------------
# Per-track stats
# ---------------------------

@dataclass
class TrackStats:
    track_id: int
    first_ts: Optional[str] = None
    last_ts: Optional[str] = None
    created: int = 0
    committed: int = 0
    expired: int = 0
    suppressed: int = 0
    count_updates: int = 0
    errors: int = 0
    classifications: int = 0
    unknown_classifications: int = 0
    label_changes: int = 0
    last_label: Optional[str] = None
    labels: Counter = field(default_factory=Counter)
    rejection_reasons: Counter = field(default_factory=Counter)
    backpressure_drops: int = 0

def update_first_last(ts: Optional[datetime], tr: TrackStats):
    if ts is None:
        return
    s = ts.isoformat().replace("+00:00", "Z")
    if tr.first_ts is None or s < tr.first_ts:
        tr.first_ts = s
    if tr.last_ts is None or s > tr.last_ts:
        tr.last_ts = s

# ---------------------------
# Streaming lines
# ---------------------------

def stream_lines(files: List[Path]) -> Iterator[Tuple[str, Path, int]]:
    for fp in files:
        with fp.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, start=1):
                line = line.rstrip("\n")
                if line.strip():
                    yield line, fp, i

# ---------------------------
# Analysis
# ---------------------------

def analyze(
    files: List[Path],
    start: Optional[datetime],
    end: Optional[datetime],
    trace_track_id: Optional[int],
    trace_event_id: Optional[int],
) -> Dict[str, Any]:
    totals = Counter()
    by_level = Counter()
    by_type = Counter()

    malformed_ts = 0
    time_filtered_out = 0

    per_minute = defaultdict(lambda: Counter())

    # undercount drivers
    expired_by_state = Counter()
    suppressed_by_reason = Counter()
    backpressure_by_queue = Counter()
    backpressure_drops_total = 0

    # classification
    label_counts = Counter()
    rejection_counts = Counter()
    classification_total = 0
    unknown_total = 0

    # counting
    count_update_total = 0
    duplicate_track_ids_in_count_updates = Counter()

    # errors
    error_lines = 0

    tracks: Dict[int, TrackStats] = {}

    trace_entries: List[Dict[str, Any]] = []

    for line, fp, line_no in stream_lines(files):
        totals["total_lines"] += 1

        ts = parse_ts_from_line(line)
        if ts is None:
            malformed_ts += 1
        else:
            if start and ts < start:
                time_filtered_out += 1
                continue
            if end and ts > end:
                time_filtered_out += 1
                continue

        level = parse_level_from_line(line)
        by_level[level] += 1

        typ = detect_type(line, level)
        by_type[typ] += 1

        minute = floor_minute(ts) if ts else None
        if minute:
            per_minute[minute][typ] += 1
            per_minute[minute][level] += 1

        event_id, track_id, frame_index = extract_ids(line)

        if (trace_track_id is not None and track_id == trace_track_id) or (
            trace_event_id is not None and event_id == trace_event_id
        ):
            trace_entries.append({
                "timestamp": ts.isoformat().replace("+00:00", "Z") if ts else None,
                "file": str(fp),
                "line": line_no,
                "level": level,
                "type": typ,
                "event_id": event_id,
                "track_id": track_id,
                "frame_index": frame_index,
                "raw": line,
            })

        # track stats
        if track_id is not None:
            tr = tracks.get(track_id)
            if tr is None:
                tr = TrackStats(track_id=track_id)
                tracks[track_id] = tr
            update_first_last(ts, tr)

            if typ == "ERROR":
                tr.errors += 1
            elif typ == "EVENT_CREATED":
                tr.created += 1
            elif typ == "EVENT_COMMITTED":
                tr.committed += 1
            elif typ == "EVENT_EXPIRED":
                tr.expired += 1
            elif typ == "EVENT_SUPPRESSED":
                tr.suppressed += 1
            elif typ == "COUNT_UPDATE":
                tr.count_updates += 1
            elif typ == "CLASSIFICATION":
                tr.classifications += 1

        # expired state
        if typ == "EVENT_EXPIRED":
            st = extract_kv_simple(EXPIRED_STATE_PAT, line) or "unknown_state"
            expired_by_state[st] += 1

        # suppression reason
        if typ == "EVENT_SUPPRESSED":
            rs = extract_kv_simple(SUPPRESSION_REASON_PAT, line) or "unknown_reason"
            suppressed_by_reason[rs] += 1

        # backpressure
        if typ == "BACKPRESSURE":
            q = extract_kv_simple(BACKPRESSURE_QUEUE_PAT, line) or "unknown_queue"
            backpressure_by_queue[q] += 1
            d = extract_kv_simple(BACKPRESSURE_DROPS_PAT, line)
            if d is not None:
                try:
                    backpressure_drops_total += int(d)
                except Exception:
                    pass
                if track_id is not None and track_id in tracks:
                    try:
                        tracks[track_id].backpressure_drops += int(d)
                    except Exception:
                        pass

        # classification
        if typ == "CLASSIFICATION":
            classification_total += 1
            label, rr, is_unknown = extract_classification(line)
            if label is None:
                label = "Unknown" if is_unknown else "unparsed_label"
            label_counts[label] += 1

            if is_unknown:
                unknown_total += 1
                rejection_counts[rr or "unknown_rejection_reason"] += 1

            if track_id is not None and track_id in tracks:
                tr = tracks[track_id]
                tr.labels[label] += 1
                if is_unknown:
                    tr.unknown_classifications += 1
                    tr.rejection_reasons[rr or "unknown_rejection_reason"] += 1
                if tr.last_label is None:
                    tr.last_label = label
                else:
                    if label != tr.last_label:
                        tr.label_changes += 1
                        tr.last_label = label

        # count updates - detect duplicates by track_id
        if typ == "COUNT_UPDATE":
            count_update_total += 1
            if track_id is not None:
                duplicate_track_ids_in_count_updates[track_id] += 1

        if typ == "ERROR":
            error_lines += 1

    # rank tracks by heuristic issue score
    track_rows = []
    for tid, tr in tracks.items():
        unknown_rate = (tr.unknown_classifications / tr.classifications) if tr.classifications else 0.0
        score = (
            tr.expired * 5 +
            tr.errors * 4 +
            tr.suppressed * 2 +
            tr.label_changes * 1 +
            int(unknown_rate * 10) +
            int(tr.backpressure_drops / 10)
        )
        track_rows.append({
            "track_id": tid,
            "issue_score": score,
            "first_ts": tr.first_ts,
            "last_ts": tr.last_ts,
            "created": tr.created,
            "committed": tr.committed,
            "expired": tr.expired,
            "suppressed": tr.suppressed,
            "count_updates": tr.count_updates,
            "errors": tr.errors,
            "classifications": tr.classifications,
            "unknown_classifications": tr.unknown_classifications,
            "unknown_rate": round(unknown_rate, 4),
            "label_changes": tr.label_changes,
            "label_diversity": len(tr.labels),
            "top_labels": tr.labels.most_common(3),
            "top_rejection_reasons": tr.rejection_reasons.most_common(3),
            "backpressure_drops": tr.backpressure_drops,
        })
    track_rows.sort(key=lambda r: r["issue_score"], reverse=True)

    # time series
    per_minute_out = []
    for minute in sorted(per_minute.keys()):
        per_minute_out.append({
            "minute": minute.isoformat().replace("+00:00", "Z"),
            "counts": dict(per_minute[minute]),
        })

    created_total = by_type["EVENT_CREATED"]
    committed_total = by_type["EVENT_COMMITTED"]
    expired_total = by_type["EVENT_EXPIRED"]

    summary: Dict[str, Any] = {
        "input": {
            "files": [str(p) for p in files],
            "time_range": {
                "from": start.isoformat().replace("+00:00", "Z") if start else None,
                "to": end.isoformat().replace("+00:00", "Z") if end else None,
            },
        },
        "parsing": {
            "total_lines": totals["total_lines"],
            "unparsed_timestamps": malformed_ts,
            "time_filtered_out": time_filtered_out,
        },
        "by_level": dict(by_level),
        "by_type": dict(by_type),
        "undercount": {
            "events_created": created_total,
            "events_committed": committed_total,
            "events_expired": expired_total,
            "commit_rate_vs_created": round((committed_total / created_total) if created_total else 0.0, 4),
            "expiration_rate_vs_created": round((expired_total / created_total) if created_total else 0.0, 4),
            "expired_by_state": dict(expired_by_state),
            "suppressed_total": by_type["EVENT_SUPPRESSED"],
            "suppressed_by_reason": dict(suppressed_by_reason),
            "backpressure_events": by_type["BACKPRESSURE"],
            "backpressure_drops_total": backpressure_drops_total,
            "backpressure_by_queue": dict(backpressure_by_queue),
            "errors_total": error_lines,
        },
        "classification": {
            "classifications_total": classification_total,
            "unknown_total": unknown_total,
            "unknown_rate": round((unknown_total / classification_total) if classification_total else 0.0, 4),
            "labels_top30": label_counts.most_common(30),
            "unknown_rejection_reasons_top30": rejection_counts.most_common(30),
        },
        "counting": {
            "count_update_total": count_update_total,
            "duplicate_track_ids_in_count_updates_top50": [
                {"track_id": tid, "count_updates": c}
                for tid, c in duplicate_track_ids_in_count_updates.most_common(50)
                if c > 1
            ],
        },
        "tracks": {
            "total_tracks_seen": len(tracks),
            "top_200": track_rows[:200],
        },
        "time_series": {
            "per_minute": per_minute_out,
        }
    }

    if trace_track_id is not None or trace_event_id is not None:
        summary["trace"] = {
            "trace_track_id": trace_track_id,
            "trace_event_id": trace_event_id,
            "entries": trace_entries,
        }

    return summary

def render_markdown(summary: Dict[str, Any]) -> str:
    p = summary["parsing"]
    u = summary["undercount"]
    c = summary["classification"]

    lines = []
    lines.append("# BreadBagCounterSystem - Deep Log Analysis (Text Logs)\n\n")
    lines.append("## Parsing\n")
    lines.append(f"- Total lines: **{p['total_lines']}**\n")
    lines.append(f"- Unparsed timestamps: **{p['unparsed_timestamps']}**\n")
    lines.append(f"- Time filtered out: **{p['time_filtered_out']}**\n")

    lines.append("\n## Under-count analysis\n")
    lines.append(f"- EVENT_CREATED: **{u['events_created']}**\n")
    lines.append(f"- EVENT_COMMITTED: **{u['events_committed']}** (commit rate: **{u['commit_rate_vs_created']*100:.2f}%**)\n")
    lines.append(f"- EVENT_EXPIRED: **{u['events_expired']}** (expiration rate: **{u['expiration_rate_vs_created']*100:.2f}%**)\n")
    lines.append(f"- EVENT_SUPPRESSED: **{u['suppressed_total']}**\n")
    lines.append(f"- BACKPRESSURE: **{u['backpressure_events']}**, drops total: **{u['backpressure_drops_total']}**\n")
    lines.append(f"- ERROR lines: **{u['errors_total']}**\n")

    lines.append("\n### Expired by state\n")
    for st, cnt in sorted(u["expired_by_state"].items(), key=lambda x: x[1], reverse=True):
        lines.append(f"- {st}: {cnt}\n")

    lines.append("\n### Suppressed by reason\n")
    for rs, cnt in sorted(u["suppressed_by_reason"].items(), key=lambda x: x[1], reverse=True):
        lines.append(f"- {rs}: {cnt}\n")

    lines.append("\n## Classification analysis\n")
    lines.append(f"- CLASSIFICATION total: **{c['classifications_total']}**\n")
    lines.append(f"- Unknown total: **{c['unknown_total']}** (unknown rate: **{c['unknown_rate']*100:.2f}%**)\n")

    lines.append("\n### Top labels\n")
    for label, cnt in c["labels_top30"][:15]:
        lines.append(f"- {label}: {cnt}\n")

    lines.append("\n### Unknown rejection reasons\n")
    for rr, cnt in c["unknown_rejection_reasons_top30"][:15]:
        lines.append(f"- {rr}: {cnt}\n")

    lines.append("\n## Track drilldown (top 25 by issue score)\n")
    lines.append("| track_id | score | created | committed | expired | suppressed | errors | classif | unknown_rate | label_changes |\n")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in summary["tracks"]["top_200"][:25]:
        lines.append(
            f"| {r['track_id']} | {r['issue_score']} | {r['created']} | {r['committed']} | "
            f"{r['expired']} | {r['suppressed']} | {r['errors']} | {r['classifications']} | "
            f"{r['unknown_rate']:.2f} | {r['label_changes']} |\n"
        )

    lines.append("\n## Notes / limitations\n")
    lines.append("- This parser is **heuristic** because `app.log` is not structured. If your format differs, we can tune regexes.\n")
    lines.append("- Timestamp handling: if your `app.log` timestamps are local time (not UTC), tell me your timezone and I’ll adjust.\n")

    return "".join(lines)

def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--output", default="analysis_out_text")
    ap.add_argument("--day", default=None, help="YYYY-MM-DD (assumed UTC day boundaries unless you tell me otherwise)")
    ap.add_argument("--from", dest="from_ts", default=None, help="ISO8601 e.g. 2026-01-06T08:00:00Z")
    ap.add_argument("--to", dest="to_ts", default=None, help="ISO8601 e.g. 2026-01-06T10:00:00Z")
    ap.add_argument("--trace-track-id", type=int, default=None)
    ap.add_argument("--trace-event-id", type=int, default=None)
    ap.add_argument("--base-name", default="app.log", help="Base text log filename (default app.log)")
    return ap.parse_args(argv)

def parse_iso_or_none(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None

def main(argv: List[str]) -> int:
    args = parse_args(argv)
    log_dir = Path(args.log_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    start: Optional[datetime] = None
    end: Optional[datetime] = None

    if args.day:
        d = datetime.fromisoformat(args.day).replace(tzinfo=timezone.utc)
        start = d
        end = d + timedelta(days=1) - timedelta(microseconds=1)

    if args.from_ts:
        start = parse_iso_or_none(args.from_ts)
        if start is None:
            print(f"Invalid --from: {args.from_ts}", file=sys.stderr)
            return 2
    if args.to_ts:
        end = parse_iso_or_none(args.to_ts)
        if end is None:
            print(f"Invalid --to: {args.to_ts}", file=sys.stderr)
            return 2

    files = discover_log_files(log_dir, base_name=args.base_name)
    if not files:
        print(f"No log files found in {log_dir} matching {args.base_name}*", file=sys.stderr)
        return 2

    print(f"Found {len(files)} log file(s):")
    for f in files:
        print(f"  - {f}")

    summary = analyze(
        files=files,
        start=start,
        end=end,
        trace_track_id=args.trace_track_id,
        trace_event_id=args.trace_event_id,
    )

    json_path = out_dir / "deep_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {json_path}")

    md_path = out_dir / "deep_summary.md"
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote {md_path}")

    csv_path = out_dir / "tracks_top.csv"
    rows = summary["tracks"]["top_200"]
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote {csv_path}")

    if args.trace_track_id is not None or args.trace_event_id is not None:
        trace = summary.get("trace", {}).get("entries", [])
        if trace:
            suffix = f"track_{args.trace_track_id}" if args.trace_track_id is not None else f"event_{args.trace_event_id}"
            trace_path = out_dir / f"trace_{suffix}.jsonl"
            with trace_path.open("w", encoding="utf-8") as f:
                for e in trace:
                    f.write(json.dumps(e, default=str) + "\n")
            print(f"Wrote {trace_path} ({len(trace)} entries)")

    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))