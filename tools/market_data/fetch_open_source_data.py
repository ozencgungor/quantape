#!/usr/bin/env python3
"""Mirror open-source example market data under data/open_source/.

Sources (commit-pinned, downloaded from raw.githubusercontent.com):

    strata    OpenGamma/Strata      example calibration resources, loader test
                                    CSVs, interest-rate FpML samples (Apache-2.0)
    ore       OpenSourceRisk/Engine  all Examples/**/Input market/fixings/trade
                                    files (BSD-3-Clause-style)
    financepy domokane/FinancePy     swaption vol surface example (GPL-3.0,
                                    local test use only)

Writes data/open_source/<source>/<repo path> and
data/open_source/manifest.json with url/sha256/size per file. Files with
identical content are stored once (first path wins); later occurrences are
recorded as aliases. Existing files are overwritten; nothing is deleted.

Usage:
    python3 tools/market_data/fetch_open_source_data.py
"""
import datetime as dt
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request

STRATA_COMMIT = "a712bfd58c650f24e180941de51bdff8fe371f98"
ORE_COMMIT = "3d75a69087911a7e2cf1e6882da7bc27135c0762"
FINANCEPY_COMMIT = "68b95fa60077eb36d965434428225f310744e14d"
QUANTLIB_COMMIT = "0191ca7cccbb62e538fd10f5c88f640594d4eaed"

QUANTLIB_FILES = (
    "test-suite/swaptionvolatilitycube.cpp",
    "test-suite/swaptionvolatilitymatrix.cpp",
    "test-suite/swaptionvolstructuresutilities.hpp",
    "test-suite/bermudanswaption.cpp",
    "test-suite/swaption.cpp",
    "test-suite/capfloor.cpp",
)

DEST_ROOT = "data/open_source"

SOURCES = [
    {
        "name": "strata",
        "repo": "OpenGamma/Strata",
        "commit": STRATA_COMMIT,
        "license": ("Apache-2.0; Copyright (C) 2016 - present by OpenGamma Inc. "
                    "and the OpenGamma group of companies"),
    },
    {
        "name": "ore",
        "repo": "OpenSourceRisk/Engine",
        "commit": ORE_COMMIT,
        "license": ("BSD-3-Clause-style; Copyright (C) 2016-2020 Quaternion Risk "
                    "Management Ltd, Copyright (C) 2021 Acadia Inc."),
    },
    {
        "name": "financepy",
        "repo": "domokane/FinancePy",
        "commit": FINANCEPY_COMMIT,
        "license": "GPL-3.0 (local test use only, do not redistribute)",
    },
    {
        "name": "quantlib",
        "repo": "lballabio/QuantLib",
        "commit": QUANTLIB_COMMIT,
        "license": ("QuantLib license (BSD-style), "
                    "https://www.quantlib.org/license.shtml"),
    },
]


def include(source, path):
    name = source["name"]
    if name == "strata":
        if path.startswith("examples/src/main/resources/"):
            return True
        if path in (
            "examples/src/main/java/com/opengamma/strata/examples/finance/SwaptionCubeData.java",
            "examples/src/main/java/com/opengamma/strata/examples/finance/SabrSwaptionCubeCalibrationExample.java",
            "LICENSE.txt", "NOTICE.txt",
        ):
            return True
        if path.startswith("modules/loader/src/test/resources/com/opengamma/strata/loader/csv/"):
            return True
        if path.startswith("modules/loader/src/test/resources/com/opengamma/strata/loader/fpml/"):
            return "swaption" in path or path.rsplit("/", 1)[-1].startswith("ird-ex")
        return False
    if name == "ore":
        if path in ("license.txt", "README.md"):
            return True
        return (path.startswith("Examples/") and "/Input/" in path
                and path.lower().endswith((".txt", ".csv", ".xml")))
    if name == "financepy":
        return path in ("LICENSE",
                        "examples/scripts/rates/example_swaption_vol_surface.py")
    if name == "quantlib":
        return path in QUANTLIB_FILES or path.upper().startswith("LICEN")
    return False


def github_tree(repo, commit):
    url = f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1"
    request = urllib.request.Request(url, headers={"User-Agent": "quantape data mirror"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if "tree" not in payload:
        raise RuntimeError(f"tree fetch failed for {repo}: {payload.get('message')}")
    return [item for item in payload["tree"] if item["type"] == "blob"]


def raw_url(repo, commit, path):
    return (f"https://raw.githubusercontent.com/{repo}/{commit}/"
            + urllib.parse.quote(path))


def fetch(url, retries=2):
    last = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "quantape data mirror"})
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001 - retry then report
            last = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"fetch failed: {url}: {last}")


def write_bytes(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "wb") as handle:
        handle.write(payload)
    os.replace(temporary, path)


def main():
    manifest_sources = []
    for source in SOURCES:
        tree = github_tree(source["repo"], source["commit"])
        selected = sorted((item for item in tree if include(source, item["path"])),
                          key=lambda item: item["path"])
        seen = {}
        files = []
        downloaded = 0
        for item in selected:
            path = item["path"]
            url = raw_url(source["repo"], source["commit"], path)
            local = os.path.join(DEST_ROOT, source["name"], path)
            if os.path.exists(local) and os.path.getsize(local) == item.get("size"):
                with open(local, "rb") as handle:
                    payload = handle.read()
            else:
                payload = fetch(url)
            digest = hashlib.sha256(payload).hexdigest()
            if digest in seen:
                files.append({"path": path, "url": url, "sha256": digest,
                              "size": len(payload), "duplicate_of": seen[digest]})
                continue
            seen[digest] = path
            write_bytes(os.path.join(DEST_ROOT, source["name"], path), payload)
            files.append({"path": path, "url": url, "sha256": digest,
                          "size": len(payload)})
            downloaded += 1
            time.sleep(0.05)
        manifest_sources.append({
            "name": source["name"],
            "repository": f"https://github.com/{source['repo']}",
            "commit": source["commit"],
            "license": source["license"],
            "n_files": len(files),
            "n_unique": downloaded,
            "bytes_unique": sum(f["size"] for f in files if "duplicate_of" not in f),
            "files": files,
        })
        print(f"{source['name']}: {downloaded} unique of {len(files)} files")

    manifest = {
        "collected_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "destination": DEST_ROOT,
        "notes": ("Third-party open-source example data for local market-data "
                  "structure design and tests. Keep license notices with the data."),
        "sources": manifest_sources,
    }
    path = os.path.join(DEST_ROOT, "manifest.json")
    write_bytes(path, json.dumps(manifest, indent=1).encode("utf-8"))
    print(f"manifest: {path}")


if __name__ == "__main__":
    main()
