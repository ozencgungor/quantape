#!/usr/bin/env python3
"""Harvest option chains every N minutes during US market hours.

Runs as a long-lived service: waits for the next US market open, then polls
the option chain for a symbol list every --interval seconds until the close
(optionally repeating daily). Source is selectable: --source cboe (one request
per root, all expiries, delayed) or --source tradier (ORATS greeks/IVs, nearest
--max-expiries expiries, sandbox delayed). Data goes through the matching
normalizer (fetch_cboe_chains.py / fetch_tradier_chains.py):

    data/cboe/{ROOT}/{YYYYMMDD-HHMMSS}.json.gz    raw payload  (--no-raw to skip)
    data/cboe/{ROOT}/{YYYYMMDD-HHMMSS}.jsonl.gz   normalized usable quotes
    data/cboe/{ROOT}/latest.jsonl                 latest snapshot
    data/cboe/manifest.jsonl                      append-only index
    data/logs/harvest-YYYYMMDD.log                service log
    data/harvest_status.json                      heartbeat (last/next poll)

Default universe (~22 roots) mixes indices, liquid megacaps, names with
earnings in the near term (vol surfaces move pre-earnings) and illiquid /
high-dispersion names (harder calibration targets):

    indices:   _SPX _VIX _NDX _RUT
    liquid:    AAPL MSFT NVDA AMZN GOOGL META TSLA JPM
    earnings:  NFLX MU NKE GS DAL
    illiquid:  GME AMC PLTR COIN SMCI

Examples:
    # one immediate snapshot (test)
    python3 harvest_cboe.py --now --once --symbols AAPL,_VIX --outdir /tmp/h
    # run through the next session and repeat daily
    python3 harvest_cboe.py --start-at-open --repeat-daily
"""
import argparse
import datetime as dt
import json
import os
import signal
import sys
import time
from zoneinfo import ZoneInfo

fetch = None

DEFAULT_SYMBOLS = [
    "_SPX", "_VIX", "_NDX", "_RUT",
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM",
    "NFLX", "MU", "NKE", "GS", "DAL",
    "GME", "AMC", "PLTR", "COIN", "SMCI",
]

STOP = False


def handle_signal(signum, frame):  # noqa: ARG001
    global STOP
    STOP = True


def parse_hhmm(text):
    hours, minutes = text.split(":")
    return dt.time(int(hours), int(minutes))


def next_open(now, tz, open_time, close_time):
    """Next trading-open datetime strictly after `now` (skips weekends)."""
    for offset in range(8):
        day = (now + dt.timedelta(days=offset)).date()
        if day.weekday() >= 5:  # Sat/Sun
            continue
        candidate = dt.datetime.combine(day, open_time, tzinfo=tz)
        if candidate > now:
            return candidate
    raise RuntimeError("no market open found")


def poll_times(day, tz, open_time, close_time, interval):
    """Open..close inclusive poll instants (open, open+interval, ...)."""
    start = dt.datetime.combine(day, open_time, tzinfo=tz)
    end = dt.datetime.combine(day, close_time, tzinfo=tz)
    times = []
    current = start
    while current <= end:
        times.append(current)
        current += dt.timedelta(seconds=interval)
    return times


def write_status(path, **fields):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(fields, handle, indent=2)
    os.replace(tmp, path)


def harvest_once(symbols, args, tz, log):
    now = dt.datetime.now(tz)
    collected = 0
    usable = 0
    for i, symbol in enumerate(symbols):
        if STOP:
            break
        try:
            manifest = fetch.collect_symbol(symbol, args)
            collected += 1
            usable += manifest["n_usable"]
            log.write(f"[{now.isoformat(timespec='seconds')}] {manifest['root']:>6}: "
                      f"{manifest['n_usable']:>5}/{manifest['n_records']:<5} usable "
                      f"(snapshot {manifest['snapshot_timestamp']})\n")
        except Exception as exc:  # noqa: BLE001 - keep going
            log.write(f"[{now.isoformat(timespec='seconds')}] ERROR {symbol}: {exc}\n")
        log.flush()
        if i + 1 < len(symbols):
            time.sleep(args.symbol_pause)
    return collected, usable


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbols", default="", help="override the default universe")
    parser.add_argument("--source", choices=("cboe", "tradier"), default="cboe")
    parser.add_argument("--outdir", default="", help="default: data/<source>")
    parser.add_argument("--status", default="", help="heartbeat file (default per source)")
    parser.add_argument("--logdir", default="data/logs")
    parser.add_argument("--interval", type=float, default=300.0, help="poll interval seconds")
    parser.add_argument("--open", default="09:30")
    parser.add_argument("--close", default="16:00")
    parser.add_argument("--tz", default="America/New_York")
    parser.add_argument("--now", action="store_true", help="start polling immediately")
    parser.add_argument("--start-at-open", action="store_true", help="wait for next market open")
    parser.add_argument("--once", action="store_true", help="single snapshot then exit")
    parser.add_argument("--repeat-daily", action="store_true", help="continue next sessions")
    parser.add_argument("--no-raw", action="store_true", help="skip raw payload files")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--symbol-pause", type=float, default=1.0)
    parser.add_argument("--cred-file", default="data/tradier/credentials.sh")
    parser.add_argument("--max-expiries", type=int, default=8,
                        help="tradier only: nearest N expiries per root (0 = all)")
    parser.add_argument("--monthly-only", action="store_true",
                        help="tradier only: third-Friday expiries only")
    parser.add_argument("--keep-incomplete", action="store_true",
                        help="tradier only: keep one-sided quotes")
    args = parser.parse_args()

    global fetch
    if args.source == "tradier":
        import fetch_tradier_chains as fetch
    else:
        import fetch_cboe_chains as fetch
    if not args.outdir:
        args.outdir = os.path.join("data", args.source)
    if not args.status:
        base = os.path.dirname(os.path.normpath(args.outdir))
        suffix = "" if args.source == "cboe" else "_" + args.source
        args.status = os.path.join(base, f"harvest_status{suffix}.json")
    if args.max_expiries == 0:
        args.max_expiries = None

    symbols = [s.strip() for s in (args.symbols.split(",") if args.symbols else DEFAULT_SYMBOLS)
               if s.strip()]
    tz = ZoneInfo(args.tz)
    open_time = parse_hhmm(args.open)
    close_time = parse_hhmm(args.close)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    log_suffix = "" if args.source == "cboe" else "-" + args.source
    log_path = os.path.join(args.logdir, f"harvest{log_suffix}-{dt.datetime.now(tz):%Y%m%d}.log")
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"=== harvest start {dt.datetime.now(tz).isoformat(timespec='seconds')} "
                  f"symbols={len(symbols)} interval={args.interval:g}s "
                  f"raw={'no' if args.no_raw else 'yes'}\n")
        log.flush()
        while not STOP:
            if args.now or args.once:
                start = dt.datetime.now(tz)
            elif args.start_at_open:
                start = next_open(dt.datetime.now(tz), tz, open_time, close_time)
                log.write(f"waiting for market open at {start.isoformat(timespec='seconds')}\n")
                log.flush()
                while not STOP and dt.datetime.now(tz) < start:
                    time.sleep(min(30.0, (start - dt.datetime.now(tz)).total_seconds()))
            else:
                start = dt.datetime.now(tz)

            if args.once:
                collected, usable = harvest_once(symbols, args, tz, log)
                log.write(f"once: {collected}/{len(symbols)} roots, {usable} usable quotes\n")
                break

            day = start.date()
            schedule = poll_times(day, tz, open_time, close_time, args.interval)
            log.write(f"session {day}: {len(schedule)} polls "
                      f"({schedule[0].isoformat(timespec='minutes')}.."
                      f"{schedule[-1].isoformat(timespec='minutes')})\n")
            log.flush()
            status_path = args.status
            for index, poll in enumerate(schedule):
                if STOP:
                    break
                now = dt.datetime.now(tz)
                if now < poll:
                    while not STOP and dt.datetime.now(tz) < poll:
                        time.sleep(min(10.0, (poll - dt.datetime.now(tz)).total_seconds()))
                if STOP:
                    break
                collected, usable = harvest_once(symbols, args, tz, log)
                next_poll = (schedule[index + 1].isoformat(timespec="seconds")
                             if index + 1 < len(schedule) else None)
                write_status(status_path,
                             last_poll=dt.datetime.now(tz).isoformat(timespec="seconds"),
                             last_session=str(day), roots_ok=collected, roots_total=len(symbols),
                             usable_quotes=usable, next_poll=next_poll)
            if not args.repeat_daily:
                break
        log.write(f"=== harvest stop {dt.datetime.now(tz).isoformat(timespec='seconds')}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
