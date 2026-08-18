"""Phase 0, step 2: download the small tabular files and audit them blind.

"Blind" is the point. This script assumes nothing about column names, the
number of targets, or which file holds the labels. It discovers them:

  * binary-looking columns          -> candidate finding targets + positive rates
  * columns with few non-null rows  -> candidate gold-label subset
  * long free-text columns          -> report text: language mix, length spread
  * low-cardinality string columns  -> candidate site / scanner / manufacturer
  * shared column names across files-> join keys, and how much they overlap

Nothing patient-derived is written to the report: no UIDs, no text samples,
only aggregate statistics. The raw JSON under artifacts/ is gitignored anyway.

Downloads only files under --max-mb with a tabular extension. DICOMs are never
touched here.

Usage:
    python eda/phase0_02_audit_tabular.py
    python eda/phase0_02_audit_tabular.py --max-mb 500
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "artifacts" / "phase0"
DATA_DIR = REPO_ROOT / "data"
COMPETITION = "rsna-knee-abnormality-detection"

TABULAR_EXT = {".csv", ".tsv", ".json", ".jsonl", ".txt", ".parquet"}
TEXT_MIN_MEAN_LEN = 40      # mean chars above which a column is treated as prose
CATEGORICAL_MAX_CARD = 60   # distinct values below which a column is a category


# --------------------------------------------------------------------------- #
# column classification
# --------------------------------------------------------------------------- #
def classify_column(series) -> dict:
    """Describe one column without ever echoing its contents."""
    import pandas as pd

    n = len(series)
    non_null = series.notna().sum()
    out: dict[str, object] = {
        "dtype": str(series.dtype),
        "n_rows": int(n),
        "n_non_null": int(non_null),
        "null_rate": round(1 - non_null / n, 6) if n else None,
        "n_unique": int(series.nunique(dropna=True)),
        "role": "unclassified",
    }
    if non_null == 0:
        out["role"] = "empty"
        return out

    values = series.dropna()

    # binary target candidate: numeric/boolean with values in {0, 1}
    if pd.api.types.is_bool_dtype(values) or pd.api.types.is_numeric_dtype(values):
        uniques = set(pd.unique(values.astype("float64")))
        if uniques <= {0.0, 1.0} and len(uniques) <= 2:
            out["role"] = "binary"
            out["positive_rate"] = round(float(values.astype("float64").mean()), 6)
            out["n_positive"] = int(values.astype("float64").sum())
            return out
        out["role"] = "numeric"
        out["min"] = float(values.min())
        out["max"] = float(values.max())
        out["mean"] = round(float(values.mean()), 6)
        return out

    text = values.astype(str)
    lengths = text.str.len()
    mean_len = float(lengths.mean())
    out["mean_char_len"] = round(mean_len, 1)

    if mean_len >= TEXT_MIN_MEAN_LEN:
        out["role"] = "free_text"
        out["len_percentiles"] = {
            f"p{p}": int(lengths.quantile(p / 100)) for p in (1, 5, 25, 50, 75, 95, 99)
        }
        out["len_max"] = int(lengths.max())
        out["languages"] = detect_languages(text)
        return out

    if out["n_unique"] == non_null:
        out["role"] = "identifier"
        # shape only, never a value
        out["example_shape"] = describe_shape(str(text.iloc[0]))
        return out

    if out["n_unique"] <= CATEGORICAL_MAX_CARD:
        out["role"] = "categorical"
        counts = text.value_counts()
        out["values"] = {str(k): int(v) for k, v in counts.head(CATEGORICAL_MAX_CARD).items()}
        return out

    out["role"] = "high_cardinality_string"
    return out


def describe_shape(value: str) -> str:
    """'1.2.826.0.1' -> 'd.d.d.d.d' — structure without the identifier."""
    out = []
    for ch in value[:60]:
        if ch.isdigit():
            out.append("d")
        elif ch.isalpha():
            out.append("a")
        else:
            out.append(ch)
    # collapse runs
    collapsed: list[str] = []
    for ch in out:
        if not collapsed or collapsed[-1] != ch:
            collapsed.append(ch)
    return "".join(collapsed)


def detect_languages(text, sample: int = 4000) -> dict:
    """Language histogram over a sample. Returns UNAVAILABLE if no detector."""
    try:
        import py3langid
    except ImportError:
        return {"_unavailable": "py3langid not installed"}
    values = text.sample(min(sample, len(text)), random_state=0) if len(text) > sample else text
    counts: Counter[str] = Counter()
    for value in values:
        snippet = value.strip()
        if len(snippet) < 15:
            counts["<too-short>"] += 1
            continue
        counts[py3langid.classify(snippet[:2000])[0]] += 1
    total = sum(counts.values())
    return {
        "n_sampled": total,
        "distribution": {k: round(v / total, 4) for k, v in counts.most_common()},
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def load_table(path: Path):
    import pandas as pd

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=path.suffix == ".jsonl")
    sep = "\t" if path.suffix == ".tsv" else ","
    return pd.read_csv(path, sep=sep, low_memory=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", default=COMPETITION)
    parser.add_argument("--max-mb", type=float, default=200.0)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument(
        "--inventory",
        default=None,
        help="output of phase0_01 (defaults to <out-dir>/competition_files.csv)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    data_dir = Path(args.data_dir)

    import pandas as pd

    inventory = Path(args.inventory or (out_dir / "competition_files.csv"))
    if not inventory.exists():
        print(f"missing {inventory}. Run eda/phase0_01_auth_and_files.py first.")
        return 2

    files = pd.read_csv(inventory)
    limit = args.max_mb * 1024 * 1024
    wanted = files[
        files["name"].str.lower().str.endswith(tuple(TABULAR_EXT))
        & (files["total_bytes"] <= limit)
    ]
    print(f"{len(wanted)} tabular files under {args.max_mb} MB (of {len(files):,} total)")
    if wanted.empty:
        print("nothing to audit")
        return 1

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    local: list[Path] = []
    for name in wanted["name"]:
        target = data_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            print(f"  downloading {name}")
            api.competition_download_file(
                args.competition, name, path=str(target.parent), quiet=True
            )
            # the API may land a .zip next to the requested name
            zipped = target.parent / (target.name + ".zip")
            if zipped.exists() and not target.exists():
                import zipfile

                with zipfile.ZipFile(zipped) as zf:
                    zf.extractall(target.parent)
                zipped.unlink()
        if target.exists():
            local.append(target)
        else:
            print(f"  WARNING: {name} did not land where expected")

    report: dict[str, object] = {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "competition": args.competition,
        "files": {},
    }
    frames: dict[str, object] = {}

    for path in local:
        rel = str(path.relative_to(data_dir))
        try:
            df = load_table(path)
        except Exception as exc:  # noqa: BLE001
            report["files"][rel] = {"_error": f"{type(exc).__name__}: {exc}"}
            continue
        frames[rel] = df
        report["files"][rel] = {
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "bytes_on_disk": path.stat().st_size,
            "columns": {str(c): classify_column(df[c]) for c in df.columns},
        }

    # ---- cross-file joins -------------------------------------------------- #
    joins = []
    names = list(frames)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(map(str, frames[a].columns)) & set(map(str, frames[b].columns))
            for col in sorted(shared):
                sa = set(frames[a][col].dropna().astype(str))
                sb = set(frames[b][col].dropna().astype(str))
                if not sa or not sb:
                    continue
                joins.append(
                    {
                        "left": a,
                        "right": b,
                        "column": col,
                        "n_left": len(sa),
                        "n_right": len(sb),
                        "n_overlap": len(sa & sb),
                        "left_covered": round(len(sa & sb) / len(sa), 4),
                        "right_covered": round(len(sa & sb) / len(sb), 4),
                    }
                )
    report["joins"] = joins

    (out_dir / "step2_audit.json").write_text(json.dumps(report, indent=2, default=str))

    # ---- markdown ---------------------------------------------------------- #
    md = [
        "# Phase 0 step 2 — tabular audit",
        "",
        f"Generated {report['generated']} by `eda/phase0_02_audit_tabular.py`.",
        "Aggregates only: no identifiers and no report text appear below.",
        "",
    ]
    for rel, info in report["files"].items():
        md += [f"## `{rel}`", ""]
        if "_error" in info:
            md += [f"failed to load: {info['_error']}", ""]
            continue
        md += [f"- rows: **{info['n_rows']:,}**, columns: **{info['n_cols']}**", ""]

        binaries = {c: v for c, v in info["columns"].items() if v["role"] == "binary"}
        if binaries:
            md += [
                f"### Binary columns — {len(binaries)} found "
                "(these are the finding targets, if this is the label file)",
                "",
                "| column | non-null rows | positives | positive rate |",
                "|---|---:|---:|---:|",
            ]
            for c, v in binaries.items():
                md.append(
                    f"| `{c}` | {v['n_non_null']:,} | {v['n_positive']:,} | "
                    f"{v['positive_rate']:.4f} |"
                )
            md.append("")
            spread = {v["n_non_null"] for v in binaries.values()}
            labelled = max(spread)
            if len(spread) > 1:
                md += [
                    "**Non-null counts differ across binary columns.** That is the "
                    "signature of a partially-labelled file — check whether the small "
                    "count is the expert-labelled subset.",
                    "",
                ]
            if labelled < 0.9 * info["n_rows"]:
                md += [
                    f"**Only {labelled:,} of {info['n_rows']:,} rows carry these labels "
                    f"({labelled / info['n_rows']:.1%}).** If this is the label file, the "
                    "gold subset is this small and the remaining rows are unlabelled — "
                    "which is the weak-supervision premise, now measured rather than "
                    "assumed.",
                    "",
                ]

        texts = {c: v for c, v in info["columns"].items() if v["role"] == "free_text"}
        for c, v in texts.items():
            md += [
                f"### Free-text column `{c}`",
                "",
                f"- non-null: **{v['n_non_null']:,}** of {v['n_rows']:,} "
                f"(null rate {v['null_rate']:.4f})",
                f"- mean length {v['mean_char_len']} chars, max {v['len_max']:,}",
                f"- length percentiles: {v['len_percentiles']}",
                "",
            ]
            langs = v.get("languages", {})
            if "distribution" in langs:
                md += [
                    f"- language mix over {langs['n_sampled']:,} sampled rows:",
                    "",
                    "| language | share |",
                    "|---|---:|",
                ]
                md += [f"| `{k}` | {p:.3f} |" for k, p in langs["distribution"].items()]
                md.append("")

        cats = {c: v for c, v in info["columns"].items() if v["role"] == "categorical"}
        if cats:
            md += [
                "### Categorical columns (site / scanner / metadata candidates)",
                "",
                "| column | distinct | top values (count) |",
                "|---|---:|---|",
            ]
            for c, v in cats.items():
                top = ", ".join(f"{k}={n}" for k, n in list(v["values"].items())[:6])
                md.append(f"| `{c}` | {v['n_unique']} | {top} |")
            md.append("")

        ids = {c: v for c, v in info["columns"].items() if v["role"] == "identifier"}
        if ids:
            md += ["### Identifier columns", "", "| column | distinct | shape |", "|---|---:|---|"]
            for c, v in ids.items():
                md.append(f"| `{c}` | {v['n_unique']:,} | `{v['example_shape']}` |")
            md.append("")

    if joins:
        md += [
            "## Joins between files",
            "",
            "| left | right | key | left rows | right rows | overlap | left covered | right covered |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for j in joins:
            md.append(
                f"| `{j['left']}` | `{j['right']}` | `{j['column']}` | {j['n_left']:,} | "
                f"{j['n_right']:,} | {j['n_overlap']:,} | {j['left_covered']:.3f} | "
                f"{j['right_covered']:.3f} |"
            )
        md.append("")

    (out_dir / "step2_audit.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:60]))
    print(f"\nwrote {(out_dir / 'step2_audit.md')}")
    print(f"wrote {(out_dir / 'step2_audit.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
