#!/usr/bin/env python3
"""Replay recorded Cboe chain snapshots as a timed feed (no perturbation).

Recorded snapshots are exactly what the harvest service collects, so this
replays the real 5-minute series — useful to drive a running calibration
engine at original speed or faster.

Inputs:
    --input DIR      directory of snapshots (e.g. data/cboe) — replays all
                     ROOT/*.jsonl.gz in timestamp order, optionally filtered
                     by --roots
    --input FILE     a single normalized .jsonl (or .jsonl.gz) snapshot

Timing:
    --speed 1        preserve the recorded inter-snapshot gaps
    --speed 10       replay 10x faster (--speed 0 = as fast as possible)
    --interval N     fixed N seconds between snapshots (overrides --speed)

Output (one JSON object per line):
    {"snapshot_ts": "...", "root": "AAPL", "quotes": [ ... ]}

Examples:
    python3 replay_chains.py --input data/cboe --roots AAPL,_SPX --speed 1
    python3 replay_chains.py --input data/cboe --speed 0 --out /tmp/feed.jsonl
"""
import argparse
import datetime as dt
import gzip
import json
import os
import re
import sys
import time

STAMP_RE = re.compile(r"(\d{8}-\d{6})")


def snapshot_paths(input_path, roots):
    if os.path.isfile(input_path):
        yield input_path, ""
        return
    entries = []
    for root in sorted(os.listdir(input_path)):
        if roots and root not in roots:
            continue
        directory = os.path.join(input_path, root)
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".jsonl.gz"):
                continue
            match = STAMP_RE.search(name)
            entries.append((match.group(1) if match else name, root, os.path.join(directory, name)))
    for stamp, root, path in sorted(entries):
        yield path, f"{stamp}:{root}"


def read_snapshot(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_stamp(label):
    stamp = label.split(":")[0]
    try:
        return dt.datetime.strptime(stamp, "%Y%m%d-%H%M%S")
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="data/cboe directory or a snapshot file")
    parser.add_argument("--roots", default="", help="comma-separated roots to include")
    parser.add_argument("--speed", type=float, default=1.0, help="1 = real time, 0 = fastest")
    parser.add_argument("--interval", type=float, default=0.0,
                        help="fixed seconds between snapshots (overrides --speed)")
    parser.add_argument("--subset", type=int, default=1, help="emit every k-th quote")
    parser.add_argument("--loop", action="store_true", help="restart after the last snapshot")
    parser.add_argument("--out", default="", help="output file (default stdout)")
    args = parser.parse_args()

    roots = {r.strip().lstrip("_").upper() for r in args.roots.split(",") if r.strip()}
    out = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout

    while True:
        previous = None
        emitted = 0
        for path, label in snapshot_paths(args.input, roots):
            quotes = read_snapshot(path)
            if args.subset > 1:
                quotes = quotes[:: args.subset]
            stamp = parse_stamp(label)
            if previous is not None and stamp is not None:
                gap = max((stamp - previous).total_seconds(), 0.0)
                delay = args.interval if args.interval > 0 else (gap / args.speed if args.speed > 0
                                                                 else 0.0)
                if delay > 0:
                    time.sleep(delay)
            if stamp is not None:
                previous = stamp
            root = label.split(":")[1] if ":" in label else (
                quotes[0]["root"] if quotes else "")
            out.write(json.dumps({
                "snapshot_ts": stamp.isoformat() if stamp else label,
                "root": root,
                "n_quotes": len(quotes),
                "quotes": quotes,
            }, separators=(",", ":")) + "\n")
            out.flush()
            emitted += 1
            print(f"replayed {label}: {len(quotes)} quotes", file=sys.stderr)
        if not args.loop or emitted == 0:
            break
    if args.out:
        out.close()


if __name__ == "__main__":
    main()
