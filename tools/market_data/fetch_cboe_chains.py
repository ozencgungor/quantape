#!/usr/bin/env python3
"""Collect Cboe delayed option chains and normalize them for calibration tests.

No API key, 15-min delayed snapshots (intraday during sessions, frozen after
the close). Each run writes, per symbol:

    {outdir}/{ROOT}/{YYYYMMDD-HHMMSS}.json.gz    raw payload
    {outdir}/{ROOT}/{YYYYMMDD-HHMMSS}.jsonl.gz   normalized usable quotes
    {outdir}/{ROOT}/latest.jsonl                 most recent normalized snapshot
    {outdir}/manifest.jsonl                      append-only collection index

Normalized quote fields: root, expiry (YYYY-MM-DD), type (C/P), strike, bid,
ask, mid, spread, vendor_iv, delta, open_interest, volume, theo,
last_trade_time. Rows require bid > 0 and ask > 0.

Examples:
    # one snapshot
    python3 fetch_cboe_chains.py --symbols AAPL,TSLA,_SPX --once
    # poll every 30 minutes (10 snapshots)
    python3 fetch_cboe_chains.py --symbols _SPX --interval 1800 --max-snapshots 10
"""
import argparse
import datetime as dt
import gzip
import json
import os
import sys
import time
import urllib.request

URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
USER_AGENT = "Mozilla/5.0 (quantape data collector)"


def parse_osi(symbol):
    """OSI 21-char symbol -> (root, expiry date, 'C'/'P', strike)."""
    root = symbol[:-15].strip()
    ymd = symbol[-15:-9]
    cp = symbol[-9]
    strike = int(symbol[-8:]) / 1000.0
    expiry = dt.date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]))
    return root, expiry.isoformat(), cp, strike


def fetch(url, timeout, retries):
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - report and retry
            last = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"fetch failed: {url}: {last}")


def normalize(payload, root):
    rows = []
    for record in payload["data"]["options"]:
        bid = float(record.get("bid") or 0.0)
        ask = float(record.get("ask") or 0.0)
        if bid <= 0.0 or ask <= 0.0:
            continue
        _, expiry, cp, strike = parse_osi(record["option"])
        rows.append({
            "root": root,
            "expiry": expiry,
            "type": cp,
            "strike": strike,
            "bid": bid,
            "ask": ask,
            "bid_size": float(record.get("bid_size") or 0.0),
            "ask_size": float(record.get("ask_size") or 0.0),
            "mid": 0.5 * (bid + ask),
            "spread": ask - bid,
            "vendor_iv": float(record.get("iv") or 0.0),
            "delta": float(record.get("delta") or 0.0),
            "gamma": float(record.get("gamma") or 0.0),
            "vega": float(record.get("vega") or 0.0),
            "theta": float(record.get("theta") or 0.0),
            "rho": float(record.get("rho") or 0.0),
            "open_interest": float(record.get("open_interest") or 0.0),
            "volume": float(record.get("volume") or 0.0),
            "theo": float(record.get("theo") or 0.0),
            "last_trade_time": record.get("last_trade_time"),
        })
    return rows


def write_gzip(path, text):
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as handle:
        handle.write(text)
    os.replace(tmp, path)


def write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp, path)


def collect_symbol(symbol, args):
    root = symbol.lstrip("_").upper()
    url = URL.format(symbol=urllib.request.quote(symbol))
    payload = fetch(url, args.timeout, args.retries)
    rows = normalize(payload, root)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    directory = os.path.join(args.outdir, root)
    os.makedirs(directory, exist_ok=True)

    if not getattr(args, "no_raw", False):
        raw_path = os.path.join(directory, f"{stamp}.json.gz")
        write_gzip(raw_path, json.dumps(payload))
    norm_path = os.path.join(directory, f"{stamp}.jsonl.gz")
    text = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows)
    write_gzip(norm_path, text)
    write_atomic(os.path.join(directory, "latest.jsonl"), text)

    underlying = {k: payload["data"].get(k) for k in
                  ("symbol", "current_price", "bid", "ask", "iv30", "volume",
                   "price_change_percent", "last_trade_time")}
    manifest = {
        "collected_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "snapshot_timestamp": payload.get("timestamp"),
        "root": root,
        "url_symbol": symbol,
        "n_records": len(payload["data"]["options"]),
        "n_usable": len(rows),
        "underlying": underlying,
    }
    with open(os.path.join(args.outdir, "manifest.jsonl"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbols", required=True,
                        help="comma-separated Cboe roots, indexes with underscore (_SPX,_VIX)")
    parser.add_argument("--outdir", default="data/cboe")
    parser.add_argument("--once", action="store_true", help="single snapshot (default)")
    parser.add_argument("--interval", type=float, default=0.0,
                        help="seconds between snapshots (0 = once)")
    parser.add_argument("--max-snapshots", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--symbol-pause", type=float, default=1.0)
    parser.add_argument("--no-raw", action="store_true", help="skip raw payload files")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        parser.error("no symbols")
    snapshots = 1 if args.once or args.interval <= 0 else args.max_snapshots
    for n in range(snapshots):
        for i, symbol in enumerate(symbols):
            try:
                manifest = collect_symbol(symbol, args)
                print(f"[{manifest['collected_at']}] {manifest['root']}: "
                      f"{manifest['n_usable']}/{manifest['n_records']} usable quotes "
                      f"(snapshot {manifest['snapshot_timestamp']})", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001 - keep other symbols going
                print(f"error for {symbol}: {exc}", file=sys.stderr)
            if i + 1 < len(symbols):
                time.sleep(args.symbol_pause)
        if n + 1 < snapshots:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
