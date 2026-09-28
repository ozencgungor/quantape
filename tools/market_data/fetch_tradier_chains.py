#!/usr/bin/env python3
"""Collect Tradier sandbox option chains and normalize them for calibration tests.

Requires a Tradier token (env TRADIER_TOKEN, or --cred-file with a shell-style
`export TRADIER_TOKEN=...` line, default data/tradier/credentials.sh). Data is
delayed in the sandbox. Each run writes, per root:

    {outdir}/{ROOT}/{YYYYMMDD-HHMMSS}.json.gz    raw payload per expiry
    {outdir}/{ROOT}/{YYYYMMDD-HHMMSS}.jsonl.gz   normalized usable quotes
    {outdir}/{ROOT}/latest.jsonl                 most recent normalized snapshot
    {outdir}/manifest.jsonl                      append-only collection index

Normalized rows keep the fetch_cboe_chains.py schema
(root, expiry, type, strike, bid, ask, mid, spread, vendor_iv, delta, open_interest,
volume, ...) and add the Tradier/ORATS extras (bid/mid/ask IV, smv_vol, phi,
bid/ask sizes, exchanges, greeks timestamp, contract size, expiration type).
Rows require bid > 0 and ask > 0 unless --keep-incomplete is given.

Examples:
    # nearest 3 expiries for a few roots
    python3 fetch_tradier_chains.py --symbols AAPL,_SPX --max-expiries 3 --once
    # monthlies only, every 5 minutes, 12 snapshots
    python3 fetch_tradier_chains.py --symbols _SPX --monthly-only \
        --interval 300 --max-snapshots 12
"""
import argparse
import datetime as dt
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://sandbox.tradier.com/v1"
DEFAULT_CRED_FILE = "data/tradier/credentials.sh"

_TOKEN = None


def load_token(args):
    token = getattr(args, "token", None) or os.environ.get("TRADIER_TOKEN")
    if token:
        return token
    path = getattr(args, "cred_file", None) or DEFAULT_CRED_FILE
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                key, _, value = line.partition("=")
                if key.strip() == "TRADIER_TOKEN":
                    return value.strip().strip('"').strip("'")
    raise RuntimeError(
        "TRADIER_TOKEN is not set and was not found in the credentials file")


def get_token(args):
    global _TOKEN
    if _TOKEN is None:
        _TOKEN = load_token(args)
    return _TOKEN


def api_get(path, params, args):
    url = BASE + path + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(args.retries + 1):
        try:
            request = urllib.request.Request(url, headers={
                "Authorization": "Bearer " + get_token(args),
                "Accept": "application/json",
            })
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last = exc
            if attempt < args.retries:
                time.sleep(5.0 * (attempt + 1) if exc.code == 429 else 1.5 * (attempt + 1))
        except Exception as exc:  # noqa: BLE001 - report and retry
            last = exc
            if attempt < args.retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"tradier request failed: {path}: {last}")


def fetch_expirations(symbol, args):
    payload = api_get("/markets/options/expirations", {
        "symbol": symbol, "includeAllRoots": "true",
    }, args)
    expirations = (payload.get("expirations") or {}).get("date")
    if not expirations:
        return []
    return sorted(set(expirations))


def select_expiries(expirations, args):
    selected = expirations
    if getattr(args, "monthly_only", False):
        monthlies = [e for e in expirations
                     if dt.date.fromisoformat(e).weekday() == 4
                     and 15 <= dt.date.fromisoformat(e).day <= 21]
        if monthlies:
            selected = monthlies
    limit = getattr(args, "max_expiries", None)
    return selected[:limit] if limit else selected


def fetch_chain(symbol, expiry, args):
    payload = api_get("/markets/options/chains", {
        "symbol": symbol, "expiration": expiry, "greeks": "true",
    }, args)
    options = payload.get("options")
    return options.get("option") or [] if options else []


def fetch_underlying(symbol, args):
    payload = api_get("/markets/quotes", {"symbols": symbol}, args)
    quotes = payload.get("quotes") or {}
    quote = quotes.get("quote")
    if isinstance(quote, list):
        quote = quote[0] if quote else {}
    return quote or {}


def iso_ms(value):
    if not value:
        return None
    try:
        return dt.datetime.fromtimestamp(value / 1000.0, dt.timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def normalize(contracts, root, keep_incomplete=False):
    rows = []
    for contract in contracts:
        greeks = contract.get("greeks") or {}
        bid = float(contract.get("bid") or 0.0)
        ask = float(contract.get("ask") or 0.0)
        if bid <= 0.0 and ask <= 0.0:
            continue
        if not keep_incomplete and (bid <= 0.0 or ask <= 0.0):
            continue
        mid_iv = float(greeks.get("mid_iv") or 0.0)
        rows.append({
            "root": root,
            "expiry": contract.get("expiration_date"),
            "type": "C" if contract.get("option_type") == "call" else "P",
            "strike": float(contract.get("strike") or 0.0),
            "bid": bid,
            "ask": ask,
            "bid_size": float(contract.get("bidsize") or 0.0),
            "ask_size": float(contract.get("asksize") or 0.0),
            "mid": 0.5 * (bid + ask),
            "spread": ask - bid,
            "vendor_iv": mid_iv,
            "delta": float(greeks.get("delta") or 0.0),
            "gamma": float(greeks.get("gamma") or 0.0),
            "vega": float(greeks.get("vega") or 0.0),
            "theta": float(greeks.get("theta") or 0.0),
            "rho": float(greeks.get("rho") or 0.0),
            "phi": float(greeks.get("phi") or 0.0),
            "bid_iv": float(greeks.get("bid_iv") or 0.0),
            "mid_iv": mid_iv,
            "ask_iv": float(greeks.get("ask_iv") or 0.0),
            "smv_vol": float(greeks.get("smv_vol") or 0.0),
            "greeks_updated_at": greeks.get("updated_at"),
            "open_interest": float(contract.get("open_interest") or 0.0),
            "volume": float(contract.get("volume") or 0.0),
            "last": float(contract.get("last") or 0.0),
            "prevclose": float(contract.get("prevclose") or 0.0),
            "theo": None,
            "last_trade_time": iso_ms(contract.get("trade_date")),
            "contract_size": contract.get("contract_size"),
            "expiration_type": contract.get("expiration_type"),
            "root_symbol": contract.get("root_symbol"),
        })
    return rows


def estimate_spot(rows, expiry=None):
    """Parity spot estimate: at the strike where call and put mids are closest,
    S ~ K + (C_mid - P_mid). Used for indices with no sandbox quote (NDX/RUT)."""
    by_strike = {}
    for row in rows:
        if expiry and row["expiry"] != expiry:
            continue
        by_strike.setdefault(row["strike"], {})[row["type"]] = row["mid"]
    best = None
    for strike, sides in by_strike.items():
        if "C" in sides and "P" in sides:
            diff = sides["C"] - sides["P"]
            if best is None or abs(diff) < abs(best[1]):
                best = (strike, diff)
    return None if best is None else best[0] + best[1]


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
    keep_incomplete = getattr(args, "keep_incomplete", False)
    expirations = select_expiries(fetch_expirations(root, args), args)
    if not expirations:
        raise RuntimeError(f"no expirations for {root}")

    raw = {"source": "tradier", "symbol": root,
           "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
           "expirations": {}}
    contracts = []
    for expiry in expirations:
        chain = fetch_chain(root, expiry, args)
        raw["expirations"][expiry] = chain
        contracts.extend(chain)
        time.sleep(getattr(args, "request_pause", 0.2))
    rows = normalize(contracts, root, keep_incomplete)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    directory = os.path.join(args.outdir, root)
    os.makedirs(directory, exist_ok=True)

    if not getattr(args, "no_raw", False):
        write_gzip(os.path.join(directory, f"{stamp}.json.gz"), json.dumps(raw))
    text = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows)
    write_gzip(os.path.join(directory, f"{stamp}.jsonl.gz"), text)
    write_atomic(os.path.join(directory, "latest.jsonl"), text)

    try:
        quote = fetch_underlying(root, args)
    except Exception as exc:  # noqa: BLE001 - underlying is best effort
        quote = {}
        print(f"underlying quote failed for {root}: {exc}", file=sys.stderr)
    underlying = {key: quote.get(key) for key in
                  ("symbol", "type", "last", "bid", "ask", "prevclose", "volume",
                   "week_52_high", "week_52_low", "trade_date")}
    if underlying.get("last") is None:
        estimate = estimate_spot(rows, expirations[0] if expirations else None)
        if estimate is not None:
            underlying["last"] = estimate
            underlying["last_estimated"] = True
    manifest = {
        "collected_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "snapshot_timestamp": raw["fetched_at"],
        "source": "tradier",
        "root": root,
        "url_symbol": root,
        "n_records": len(contracts),
        "n_usable": len(rows),
        "expiries": expirations,
        "underlying": underlying,
    }
    with open(os.path.join(args.outdir, "manifest.jsonl"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbols", required=True,
                        help="comma-separated roots, Cboe-style indexes with underscore (_SPX,_VIX)")
    parser.add_argument("--outdir", default="data/tradier")
    parser.add_argument("--token", default=None, help="Tradier bearer token (default: env/cred-file)")
    parser.add_argument("--cred-file", default=DEFAULT_CRED_FILE)
    parser.add_argument("--max-expiries", type=int, default=8,
                        help="nearest N expirations per root (0 = all)")
    parser.add_argument("--monthly-only", action="store_true",
                        help="keep only third-Friday expirations")
    parser.add_argument("--keep-incomplete", action="store_true",
                        help="also keep quotes with one side missing")
    parser.add_argument("--once", action="store_true", help="single snapshot (default)")
    parser.add_argument("--interval", type=float, default=0.0,
                        help="seconds between snapshots (0 = once)")
    parser.add_argument("--max-snapshots", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--symbol-pause", type=float, default=1.0)
    parser.add_argument("--request-pause", type=float, default=0.2)
    parser.add_argument("--no-raw", action="store_true", help="skip raw payload files")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        parser.error("no symbols")
    if getattr(args, "max_expiries", 0) == 0:
        args.max_expiries = None
    os.makedirs(args.outdir, exist_ok=True)
    snapshots = 1 if args.once or args.interval <= 0 else args.max_snapshots
    for n in range(snapshots):
        for i, symbol in enumerate(symbols):
            try:
                manifest = collect_symbol(symbol, args)
                print(f"[{manifest['collected_at']}] {manifest['root']}: "
                      f"{manifest['n_usable']}/{manifest['n_records']} usable quotes "
                      f"over {len(manifest['expiries'])} expiries", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001 - keep other symbols going
                print(f"error for {symbol}: {exc}", file=sys.stderr)
            if i + 1 < len(symbols):
                time.sleep(args.symbol_pause)
        if n + 1 < snapshots:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
