#!/usr/bin/env python3
"""Extract free swaption vol-cube fixtures from cached open-source test data.

Parses the raw files under data/swaption_vol/raw/ (downloaded from the
open-source projects below; see manifest for commit pins and licenses) and
writes normalized JSON plus a provenance manifest:

    data/swaption_vol/normalized/quantlib_swaption_atm_matrix.json
    data/swaption_vol/normalized/quantlib_swaption_cube_spreads.json
    data/swaption_vol/normalized/strata_eur_swaption_cube_20160229.json
    data/swaption_vol/manifest.json

No network access. The QuantLib fixture is lognormal ATM vols plus relative
smile spreads on a 5-strike-spread grid; the Strata fixture is a real EUR
normal-vol cube (2016-02-29) with full and sparse variants, whose matching
multi-curve calibration quotes are cached under
data/swaption_vol/raw/strata_calibration/.

Usage:
    python3 tools/market_data/extract_swaption_vol.py
"""
import datetime as dt
import hashlib
import json
import os
import re

ROOT = "data/swaption_vol"
OUT = os.path.join(ROOT, "normalized")

QL_RAW = "data/open_source/quantlib/test-suite"
STRATA_JAVA = ("data/open_source/strata/examples/src/main/java/"
               "com/opengamma/strata/examples/finance")
STRATA_RESOURCES = ("data/open_source/strata/examples/src/main/resources/"
                    "example-calibration")
ORE_RAW = "data/open_source/ore/Examples/Input"
FINANCEPY_RAW = "data/open_source/financepy/examples/scripts/rates"

QUANTLIB_COMMIT = "0191ca7cccbb62e538fd10f5c88f640594d4eaed"
STRATA_COMMIT = "a712bfd58c650f24e180941de51bdff8fe371f98"
ORE_COMMIT = "3d75a69087911a7e2cf1e6882da7bc27135c0762"
FINANCEPY_COMMIT = "68b95fa60077eb36d965434428225f310744e14d"


def read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def period_label(length, unit):
    unit = unit.rstrip("s")
    return f"{length}{unit[0]}"


def parse_quantlib_matrix(text):
    text = text[text.index("struct AtmVolatility"):text.index("struct VolatilityCube")]
    options = {}
    swaps = {}
    for index, length, unit in re.findall(
            r"tenors\.options\[(\d+)\] = Period\((\d+), (\w+)\);", text):
        options[int(index)] = period_label(length, unit)
    for index, length, unit in re.findall(
            r"tenors\.swaps\[(\d+)\] = Period\((\d+), (\w+)\);", text):
        swaps[int(index)] = period_label(length, unit)
    vols = {}
    for i, j, value in re.findall(r"vols\[(\d+)\]\[(\d+)\]\s*=\s*([0-9.]+);", text):
        vols[(int(i), int(j))] = float(value)
    n_rows = len(options)
    n_cols = len(swaps)
    matrix = [[vols[(i, j)] for j in range(n_cols)] for i in range(n_rows)]
    return {
        "option_tenors": [options[i] for i in range(n_rows)],
        "swap_tenors": [swaps[j] for j in range(n_cols)],
        "atm_vols": matrix,
    }


def parse_quantlib_cube(text):
    text = text[text.index("struct VolatilityCube"):]
    options = dict((int(i), period_label(length, unit)) for i, length, unit in
                   re.findall(r"tenors\.options\[(\d+)\] = Period\((\d+), (\w+)\);", text))
    swaps = dict((int(i), period_label(length, unit)) for i, length, unit in
                 re.findall(r"tenors\.swaps\[(\d+)\] = Period\((\d+), (\w+)\);", text))
    strikes = [float(value) for _, value in
               re.findall(r"strikeSpreads\[(\d+)\] = ([-+0-9.]+);", text)]
    spreads = {}
    for i, j, value in re.findall(
            r"volSpreads\[(\d+)\]\[(\d+)\]\s*=\s*([-+0-9.]+);", text):
        spreads[(int(i), int(j))] = float(value)
    n_rows = len(options) * len(swaps)
    n_cols = len(strikes)
    matrix = [[spreads[(i, j)] for j in range(n_cols)] for i in range(n_rows)]
    return {
        "option_tenors": [options[i] for i in range(len(options))],
        "swap_tenors": [swaps[j] for j in range(len(swaps))],
        "strike_spreads": strikes,
        "strike_spreads_bp": [round(s * 10000.0, 4) for s in strikes],
        "vol_spreads": matrix,
    }


def extract_braced(text, marker):
    start = text.index(marker) + len(marker)
    start = text.index("{", start)
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise RuntimeError(f"unbalanced braces after {marker}")


def cube_from_braces(block):
    rows = re.findall(r"\{([^{}]*)\}", block)
    cube = []
    for row in rows:
        values = []
        for token in row.split(","):
            token = token.strip()
            values.append(None if token == "Double.NaN" else float(token))
        cube.append(values)
    return cube


def parse_strata_cube(text):
    moneyness = [float(v) for v in
                 re.search(r"MONEYNESS =\s*DoubleArray\.of\(([^)]*)\)", text).group(1).split(",")]
    expiries = []
    for match in re.finditer(r"EXPIRIES\.add\(Period\.of(Months|Years)\((\d+)\)\);", text):
        unit, length = match.groups()
        expiries.append(period_label(length, unit))
    tenors = re.findall(r"TENORS\.add\(Tenor\.TENOR_(\d+)Y\);", text)
    full_flat = cube_from_braces(extract_braced(text, "DATA_ARRAY_FULL ="))
    sparse_flat = cube_from_braces(extract_braced(text, "DATA_ARRAY_SPARSE ="))

    def reshape(flat):
        n_tenors = len(tenors)
        n_expiries = len(expiries)
        if len(flat) != n_tenors * n_expiries:
            raise RuntimeError(
                f"expected {n_tenors * n_expiries} rows, found {len(flat)}")
        return [flat[i * n_expiries:(i + 1) * n_expiries] for i in range(n_tenors)]

    full = reshape(full_flat)
    sparse = reshape(sparse_flat)
    return {
        "data_date": "2016-02-29",
        "currency": "EUR",
        "convention": "EUR_FIXED_1Y_EURIBOR_6M",
        "day_count": "ACT_365F",
        "vol_type": "normal",
        "moneyness_type": "simple moneyness (strike - par swap rate, rate units)",
        "expiries": expiries,
        "tenors": [f"{t}Y" for t in tenors],
        "moneyness": moneyness,
        "cube_full": full,
        "cube_sparse": sparse,
        "example_settings": {"sabr_beta": 0.50, "shift": 0.03},
    }


PERIOD_RE = re.compile(r"^(\d+)([WMY])$")


def period_months(label):
    match = PERIOD_RE.match(label)
    if not match:
        raise ValueError(f"unrecognized period: {label}")
    length, unit = int(match.group(1)), match.group(2)
    return length * {"W": 0.25, "M": 1, "Y": 12}[unit]


def parse_ore_swaption_atm(text):
    pattern = re.compile(
        r"^\d{8} SWAPTION/RATE_LNVOL/([A-Z]{3})/(\S+?)/(\S+?)/([A-Z0-9]+)\s+(-?[0-9.]+)\s*$",
        re.M)
    cells = {}
    for ccy, expiry, tenor, tag, value in pattern.findall(text):
        if tag != "ATM":
            continue
        cells.setdefault(ccy, {})[(expiry, tenor)] = float(value)
    result = {}
    for ccy in sorted(cells):
        block = cells[ccy]
        expiries = sorted({e for e, _ in block}, key=period_months)
        tenors = sorted({t for _, t in block}, key=period_months)
        result[ccy] = {
            "expiries": expiries,
            "tenors": tenors,
            "atm_lognormal_vols": [[block.get((e, t)) for t in tenors] for e in expiries],
        }
    return result


def parse_ore_capfloor(text):
    pattern = re.compile(
        r"^\d{8} CAPFLOOR/RATE_LNVOL/([A-Z]{3})/(\S+?)/(\S+?)/([^/\s]+)/([^/\s]+)/([0-9.]+)\s+([0-9.]+)\s*$",
        re.M)
    rows = []
    for ccy, expiry, tenor, param1, param2, strike, vol in pattern.findall(text):
        rows.append({"ccy": ccy, "expiry": expiry, "tenor": tenor,
                     "param1": param1, "param2": param2,
                     "strike": float(strike), "vol": float(vol)})
    by_ccy = {}
    for row in rows:
        by_ccy.setdefault(row["ccy"], []).append(row)
    return {ccy: by_ccy[ccy] for ccy in sorted(by_ccy)}


def extract_square_block(text, marker):
    start = text.index(marker)
    start = text.index("[", start)
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "[":
            depth += 1
        elif text[index] == "]":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise RuntimeError(f"unbalanced brackets after {marker}")


def matrix_from_block(block):
    rows = []
    for row in re.findall(r"\[([^\[\]]*)\]", block):
        rows.append([float(value) for value in re.findall(r"-?\d+\.?\d*", row)])
    return rows


def parse_financepy_surface(text):
    vols = [[value / 100.0 for value in row]
            for row in matrix_from_block(extract_square_block(text, "market_volatilities = ["))]
    strikes = [[value / 100.0 for value in row]
               for row in matrix_from_block(extract_square_block(text, "market_strikes = ["))]
    dates = []
    for day, month, year in re.findall(
            r"Date\((\d+), (\d+), (\d+)\)",
            extract_square_block(text, "exercise_dts = [")):
        dates.append(dt.date(int(year), int(month), int(day)).isoformat())
    value_date = re.search(r"value_dt = Date\((\d+), (\d+), (\d+)\)", text)
    if value_date:
        day, month, year = value_date.groups()
        value_date = dt.date(int(year), int(month), int(day)).isoformat()
    return {
        "value_date": value_date,
        "exercise_dates": dates,
        "fwd_swap_rates": strikes[3] if len(strikes) > 3 else None,
        "market_vols": vols,
        "market_strikes": strikes,
    }


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")
    os.replace(temporary, path)


def quantlib_url(name):
    return (f"https://github.com/lballabio/QuantLib/blob/{QUANTLIB_COMMIT}/test-suite/{name}")


def strata_url(name):
    return (f"https://github.com/OpenGamma/Strata/blob/{STRATA_COMMIT}/"
            f"examples/src/main/java/com/opengamma/strata/examples/finance/{name}")


def main():
    header = read(os.path.join(QL_RAW, "swaptionvolstructuresutilities.hpp"))
    matrix = parse_quantlib_matrix(header)
    cube = parse_quantlib_cube(header)
    strata = parse_strata_cube(read(os.path.join(STRATA_JAVA, "SwaptionCubeData.java")))
    ore_text = read(os.path.join(ORE_RAW, "market_20160205.txt"))
    ore_atm = parse_ore_swaption_atm(ore_text)
    ore_capfloor = parse_ore_capfloor(ore_text)
    financepy = parse_financepy_surface(
        read(os.path.join(FINANCEPY_RAW, "example_swaption_vol_surface.py")))

    outputs = {
        "quantlib_swaption_atm_matrix.json": {
            "provenance": {
                "project": "QuantLib",
                "file": "test-suite/swaptionvolstructuresutilities.hpp",
                "url": quantlib_url("swaptionvolstructuresutilities.hpp"),
                "commit": QUANTLIB_COMMIT,
                "license": "QuantLib license (BSD-style)",
            },
            "conventions": {
                "calendar": "TARGET",
                "business_day_convention": "ModifiedFollowing",
                "day_count": "Actual365Fixed",
                "vol_type": "lognormal (Black)",
                "quote_units": "decimal (0.1300 = 13%)",
            },
            **matrix,
        },
        "quantlib_swaption_cube_spreads.json": {
            "provenance": {
                "project": "QuantLib",
                "file": "test-suite/swaptionvolstructuresutilities.hpp",
                "url": quantlib_url("swaptionvolstructuresutilities.hpp"),
                "commit": QUANTLIB_COMMIT,
                "license": "QuantLib license (BSD-style)",
            },
            "conventions": {
                "calendar": "TARGET",
                "business_day_convention": "ModifiedFollowing",
                "day_count": "Actual365Fixed",
                "vol_type": "lognormal (Black)",
                "spread_definition": "market vol - atm vol (decimal)",
            },
            "note": ("sections are ordered option-major: "
                     "option_tenors x swap_tenors; QuantLib's cube test also builds a "
                     "derived normal-vol variant as 0.05 * atm_vols (not market data)"),
            **cube,
        },
        "strata_eur_swaption_cube_20160229.json": {
            "provenance": {
                "project": "OpenGamma Strata",
                "file": "examples/.../finance/SwaptionCubeData.java",
                "url": strata_url("SwaptionCubeData.java"),
                "commit": STRATA_COMMIT,
                "license": "Apache-2.0",
                "copyright": "Copyright (C) 2016 - present by OpenGamma Inc. and the OpenGamma group of companies",
            },
            "note": ("indices are [tenor][expiry][moneyness]; cube_sparse replaces missing "
                     "points with null; data from 29-February-2016 per the source file"),
            **strata,
        },
        "ore_swaption_atm_20160205.json": {
            "provenance": {
                "project": "Open Source Risk Engine (ORE)",
                "file": "Examples/Input/market_20160205.txt",
                "url": (f"https://github.com/OpenSourceRisk/Engine/blob/{ORE_COMMIT}/"
                        "Examples/Input/market_20160205.txt"),
                "commit": ORE_COMMIT,
                "license": ("BSD-3-Clause-style (Copyright (C) 2016-2020 Quaternion Risk "
                            "Management Ltd, Copyright (C) 2021 Acadia Inc.)"),
            },
            "conventions": {
                "data_date": "2016-02-05",
                "vol_type": "lognormal (Black)",
                "quote_units": "decimal (0.3433 = 34.33%)",
                "note": ("ORE example market data, illustrative; not guaranteed live marks; "
                         "missing maturity combinations are null"),
            },
            "data": ore_atm,
        },
        "ore_capfloor_vols_20160205.json": {
            "provenance": {
                "project": "Open Source Risk Engine (ORE)",
                "file": "Examples/Input/market_20160205.txt",
                "url": (f"https://github.com/OpenSourceRisk/Engine/blob/{ORE_COMMIT}/"
                        "Examples/Input/market_20160205.txt"),
                "commit": ORE_COMMIT,
                "license": ("BSD-3-Clause-style (Copyright (C) 2016-2020 Quaternion Risk "
                            "Management Ltd, Copyright (C) 2021 Acadia Inc.)"),
            },
            "conventions": {
                "data_date": "2016-02-05",
                "vol_type": "lognormal (Black)",
                "strike_units": "decimal (0.025 = 2.5%)",
                "note": "param1/param2 are the two 0/0 fields in the source key",
            },
            "capfloor": ore_capfloor,
        },
        "financepy_swaption_surface_example.json": {
            "provenance": {
                "project": "FinancePy",
                "file": "examples/scripts/rates/example_swaption_vol_surface.py",
                "url": (f"https://github.com/domokane/FinancePy/blob/{FINANCEPY_COMMIT}/"
                        "examples/scripts/rates/example_swaption_vol_surface.py"),
                "commit": FINANCEPY_COMMIT,
                "license": "GPL-3.0",
                "note": "local test use only; do not redistribute with the library",
            },
            "note": ("7 strikes x 8 expiries; source quotes percentages, stored /100; "
                     "strikes are per-expiry and fwd_swap_rates is strike row 3"),
            **financepy,
        },
    }
    for name, payload in outputs.items():
        write_json(os.path.join(OUT, name), payload)

    calibration = [
        "curves/EUR-DSCONOIS-E3BS-E6IRS-group.csv",
        "curves/EUR-DSCONOIS-E3BS-E6IRS-nodes.csv",
        "curves/EUR-DSCONOIS-E3BS-E6IRS-settings.csv",
        "quotes/quotes-20160229-eur.csv",
    ]
    manifest = {
        "collected_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "notes": ("Third-party open-source fixture data, cached for local calibration "
                  "and data-structure tests. Redistribution requires retaining the "
                  "license notices of the source projects."),
        "sources": [
            {
                "project": "QuantLib",
                "repository": "https://github.com/lballabio/QuantLib",
                "commit": QUANTLIB_COMMIT,
                "license": "QuantLib license (BSD-style), https://www.quantlib.org/license.shtml",
                "raw_files": {
                    "test-suite/" + name: sha256(os.path.join(QL_RAW, name))
                    for name in (
                        "swaptionvolstructuresutilities.hpp",
                        "swaptionvolatilitycube.cpp",
                        "swaptionvolatilitymatrix.cpp",
                        "bermudanswaption.cpp",
                        "swaption.cpp",
                        "capfloor.cpp",
                    )
                },
            },
            {
                "project": "OpenGamma Strata",
                "repository": "https://github.com/OpenGamma/Strata",
                "commit": STRATA_COMMIT,
                "license": "Apache-2.0",
                "raw_files": {
                    name: sha256(os.path.join(STRATA_JAVA, name)) for name in (
                        "SwaptionCubeData.java",
                        "SabrSwaptionCubeCalibrationExample.java",
                    )
                },
                "calibration_fixture": {
                    "dir": STRATA_RESOURCES,
                    "data_date": "2016-02-29",
                    "files": {name: sha256(os.path.join(STRATA_RESOURCES, name))
                              for name in calibration},
                    "description": ("EUR multi-curve quotes (OIS, 3M/6M basis, 6M IRS), "
                                    "curve nodes/settings/groups paired with the EUR cube"),
                },
            },
            {
                "project": "Open Source Risk Engine (ORE)",
                "repository": "https://github.com/OpenSourceRisk/Engine",
                "commit": ORE_COMMIT,
                "license": ("BSD-3-Clause-style, see license.txt "
                            "(Quaternion Risk Management Ltd / Acadia Inc.)"),
                "raw_files": {
                    "Examples/Input/market_20160205.txt":
                        sha256(os.path.join(ORE_RAW, "market_20160205.txt")),
                    "Examples/Legacy/Example_19/Input/market_20160205_smile.txt":
                        sha256(os.path.join("data/open_source/ore/Examples/Legacy/"
                                            "Example_19/Input/market_20160205_smile.txt")),
                },
                "note": ("smile file cached but not extracted: its SWAPTION/RATE_NVOL "
                         "values are a flat example (0.005-0.008)"),
            },
            {
                "project": "FinancePy",
                "repository": "https://github.com/domokane/FinancePy",
                "commit": FINANCEPY_COMMIT,
                "license": "GPL-3.0 (local test use only; do not redistribute)",
                "raw_files": {
                    "examples/scripts/rates/example_swaption_vol_surface.py":
                        sha256(os.path.join(FINANCEPY_RAW,
                                            "example_swaption_vol_surface.py")),
                },
            },
        ],
        "outputs": {
            name: sha256(os.path.join(OUT, name)) for name in outputs
        },
    }
    write_json(os.path.join(ROOT, "manifest.json"), manifest)
    print(f"wrote {len(outputs)} normalized fixtures + manifest under {ROOT}/")


if __name__ == "__main__":
    main()
