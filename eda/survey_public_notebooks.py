#!/usr/bin/env python3
r"""Map what the public field is actually built from, before borrowing any of it.

    kaggle kernels list --competition rsna-knee-abnormality-detection \
        --page-size 100 -p 1 -v > corpus/index.csv     # repeat for each page
    while read ref; do kaggle kernels pull "$ref" -p "corpus/nb/${ref//\//~}" -m; done
    python eda/survey_public_notebooks.py --corpus corpus/nb --index corpus/index.csv

**Why this exists.** `survey_public_checkpoints.py` screens one file once someone
has already decided it is worth downloading. Nothing screened the *field*, and
the field is where the decision actually gets made: 933 of 3,332 teams sit on
two forked notebooks (E088), so what those notebooks mount is what the whole
plateau depends on, and a single undeclared dataset underneath them puts the
entire route out of reach.

**The question it answers is dependency, not quality.** Rank every asset by how
many public notebooks mount it, then read the licence of the ones at the top.
That ordering is what says whether a plateau is reachable under E043's rule or
merely popular. It found, on 2026-09-08, that the 0.937 notebook rests on three
CC0 datasets, one Apache-2.0 set, and one licensed *"Other (specified in
description)"* whose description is empty.

**Licences are supplied, not fetched.** The screen must run offline and be
testable, and a licence read at survey time is a fact with a date on it, like a
rank — so it is passed in as JSON and recorded, never silently refreshed.

**E043's three tiers, which decide what the counts mean.**

1. **CC0 / Apache-2.0** — usable, attribution as a courtesy.
2. **CC-BY-NC-SA** — NC matches the competition's own winner licence, so NC is
   not the obstacle; **ShareAlike on derivatives is the part to read.** This is
   a "read before shipping", not an exclusion, and reading it as an exclusion is
   the error E088 caught in three standing documents.
3. **`not-declared`, or `other` with nothing specified** — no grant of any kind.
   **Avoid.** An undeclared licence is not a permissive one.

Reads no patient data. Reads only notebook source and metadata, and prints
aggregates.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from pathlib import Path

# E043's tiers, as prefixes of the licence strings Kaggle actually returns.
USABLE = ("CC0", "Apache")
SHAREALIKE = ("CC BY-NC-SA", "CC-BY-NC-SA")
NO_GRANT = ("Other", "Unknown", "not-declared", "No license", "Other (specified in description)")

# Backbone families worth counting separately. The point is not a leaderboard of
# architectures — E088 priced architecture at 0.000 every time this project
# measured it — but to see which families the plateau depends on, because a
# family nobody else uses is the one a union would actually decorrelate against.
BACKBONES = {
    "dinov3": r"dinov3",
    "dinov2": r"dinov2",
    "coatnet": r"coatnet",
    "convnext": r"convnext|cnv2|cnx",
    "efficientnet": r"efficientnet|effv2|tf_efficient",
    "radimagenet": r"radimagenet",
    "resnet": r"resnet\d",
    "swin": r"swin_|swin-",
    "vit": r"vit_[bsl]|vit-[bsl]",
    "maxvit": r"maxvit",
}


def tier(licence: str | None) -> str:
    """E043's tier for a licence string, or 'unknown' when none was supplied."""
    if not licence:
        return "unknown"
    if licence.startswith(USABLE):
        return "usable"
    if any(licence.startswith(s) for s in SHAREALIKE):
        return "sharealike"
    if licence.startswith(NO_GRANT):
        return "no-grant"
    return "unknown"


def read_notebook(directory: Path) -> dict | None:
    """Metadata plus a lowercased copy of the source, or None if the pull failed."""
    meta_path = directory / "kernel-metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None
    source = ""
    for path in directory.iterdir():
        if path.suffix in (".py", ".ipynb", ".r", ".R"):
            source += path.read_text(encoding="utf-8", errors="replace").lower()
    return {
        "ref": meta.get("id", directory.name.replace("~", "/")),
        "datasets": list(meta.get("dataset_sources") or []),
        "kernels": list(meta.get("kernel_sources") or []),
        "models": list(meta.get("model_sources") or []),
        "gpu": bool(meta.get("enable_gpu")),
        "internet": bool(meta.get("enable_internet")),
        "reads_dicom": "pydicom" in source or "dcmread" in source,
        "backbones": sorted(n for n, p in BACKBONES.items() if re.search(p, source)),
        "source_chars": len(source),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True,
                        help="directory of pulled notebooks, one subdirectory each")
    parser.add_argument("--index", help="CSV from `kaggle kernels list -v`, for vote counts")
    parser.add_argument("--licences", help="JSON mapping owner/slug to a licence string")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args(argv)

    licences = json.loads(Path(args.licences).read_text()) if args.licences else {}
    votes = {}
    if args.index:
        with open(args.index, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("ref"):
                    votes[row["ref"]] = int(row.get("totalVotes") or 0)

    dirs = sorted(p for p in Path(args.corpus).iterdir() if p.is_dir())
    books = [b for b in (read_notebook(d) for d in dirs) if b]
    print(f"notebooks read: {len(books)} of {len(dirs)} directories "
          f"({len(dirs) - len(books)} had no usable metadata)\n")
    if not books:
        return 0

    # --- asset census, weighted two ways: how many notebooks, and how much attention
    mounts = collections.Counter()
    attention = collections.Counter()
    for b in books:
        for d in b["datasets"]:
            mounts[d] += 1
            attention[d] += votes.get(b["ref"], 0)

    print(f"{'mounted by':>10} {'votes':>7}  {'tier':11s} {'licence':26s} dataset")
    for asset, n in mounts.most_common(args.top):
        lic = licences.get(asset)
        print(f"{n:>10} {attention[asset]:>7}  {tier(lic):11s} "
              f"{(lic or '—')[:26]:26s} {asset}")

    counts = collections.Counter(tier(licences.get(a)) for a in mounts)
    print("\nassets by tier: " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    if counts.get("no-grant"):
        print("  NO-GRANT assets are E043's avoid tier — an undeclared licence is "
              "not a permissive one.")
    if counts.get("sharealike"):
        print("  SHAREALIKE is a read, not an exclusion (E043). NC matches this "
              "competition's own winner licence.")

    # --- backbone census
    fam = collections.Counter()
    for b in books:
        for name in b["backbones"]:
            fam[name] += 1
    print(f"\nbackbone families named in source, of {len(books)} notebooks:")
    for name, n in fam.most_common():
        print(f"   {name:14s} {n:4d}  ({100 * n / len(books):4.1f}%)")

    dicom = sum(b["reads_dicom"] for b in books)
    print(f"\nnotebooks that read DICOM at run time: {dicom} "
          f"({100 * dicom / len(books):.1f}%) — the rest mount a preprocessed corpus")
    print("\nEVERY LICENCE ABOVE WAS SUPPLIED, NOT FETCHED. It is a fact with a "
          "date on it.\nRe-read before shipping anything that depends on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
