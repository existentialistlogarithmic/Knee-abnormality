"""Phase 0, step 1: prove Kaggle access and inventory the competition files.

What this does NOT do: download anything. It lists file names and sizes only,
so it is safe to run on a laptop against a ~570 GB competition.

Outputs (all under artifacts/phase0/, gitignored because file paths embed
StudyInstanceUIDs):
    competition_meta.json     raw competition metadata as the API reports it
    competition_files.csv     one row per file: name, bytes, creation date
    step1_summary.md          a block to paste/merge into docs/FINDINGS.md
    step1_summary_public.md   same, with nested (UID-bearing) paths redacted

Every printed number comes from the API. Nothing is inferred or filled in.

Usage:
    python eda/phase0_01_auth_and_files.py
    python eda/phase0_01_auth_and_files.py --competition some-other-comp
    python eda/phase0_01_auth_and_files.py --max-pages 5   # sample, don't page all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

COMPETITION = "rsna-knee-abnormality-detection"
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "artifacts" / "phase0"


# --------------------------------------------------------------------------- #
# credentials
# --------------------------------------------------------------------------- #
def describe_credential_source() -> str:
    """Report which credential mechanism is present. Never prints the secret."""
    home = Path(os.path.expanduser("~"))
    if os.environ.get("KAGGLE_API_TOKEN"):
        return "env KAGGLE_API_TOKEN"
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return "env KAGGLE_USERNAME + KAGGLE_KEY"
    if (home / ".kaggle" / "access_token").exists():
        return "~/.kaggle/access_token"
    if (home / ".kaggle" / "kaggle.json").exists():
        return "~/.kaggle/kaggle.json (legacy, still accepted by CLI 2.x)"
    if (home / ".cache" / "kaggle" / "auth.json").exists():
        return "OAuth cache from `kaggle auth login`"
    return "NONE FOUND"


AUTH_HELP = """
No Kaggle credentials found. Any one of these works with kaggle CLI 2.x:

  1. kaggle auth login                      (OAuth, no token file to manage)
  2. export KAGGLE_API_TOKEN=<token>        (token from kaggle.com/settings/api)
  3. write the token to ~/.kaggle/access_token
  4. legacy ~/.kaggle/kaggle.json           (chmod 600)

Then re-run this script.
"""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def human_bytes(n: int) -> str:
    step = 1024.0
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < step or unit == "TB":
            return f"{value:,.1f} {unit}"
        value /= step
    return f"{value:,.1f} TB"


def top_level(name: str) -> str:
    """First path component, or '<root>' for files sitting at the top."""
    parts = name.replace("\\", "/").split("/")
    return parts[0] if len(parts) > 1 else "<root>"


def extension(name: str) -> str:
    base = name.replace("\\", "/").split("/")[-1]
    if "." not in base:
        return "<none>"
    return "." + base.split(".", 1)[1] if base.startswith(".") else "." + base.rsplit(".", 1)[1]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", default=COMPETITION)
    parser.add_argument("--page-size", type=int, default=200, help="API max is 200")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="seconds to wait between pages; use to avoid rate limits")
    parser.add_argument("--backoff", type=float, default=5.0,
                        help="initial backoff after a 429, doubling each retry")
    parser.add_argument("--max-backoff", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="ignore any saved checkpoint and list from the start")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="0 = page through everything. Set a limit to sample a huge listing.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    source = describe_credential_source()
    print(f"credential source : {source}")
    if source == "NONE FOUND":
        print(AUTH_HELP)
        return 2

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("kaggle package not installed: pip install -r requirements.txt")
        return 2

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # noqa: BLE001 - we want the raw reason surfaced
        print(f"AUTH FAILED: {type(exc).__name__}: {exc}")
        print(AUTH_HELP)
        return 2
    print("authenticated     : yes")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- competition metadata --------------------------------------------- #
    meta: dict[str, object] = {}
    try:
        response = api.competitions_list(search=args.competition, page_size=50)
        matches = [
            c
            for c in (getattr(response, "competitions", None) or [])
            if (getattr(c, "ref", "") or "").rstrip("/").split("/")[-1] == args.competition
        ]
        if matches:
            comp = matches[0]
            for field in (
                "ref",
                "title",
                "url",
                "organization_name",
                "host_name",
                "category",
                "reward",
                "evaluation_metric",
                "deadline",
                "merger_deadline",
                "new_entrant_deadline",
                "enabled_date",
                "max_daily_submissions",
                "max_team_size",
                "is_kernels_submissions_only",
                "submissions_disabled",
                "team_count",
                "user_has_entered",
            ):
                value = getattr(comp, field, None)
                meta[field] = value.isoformat() if isinstance(value, datetime) else value
        else:
            meta["_note"] = (
                "competition not returned by search; it may be unlisted to this "
                "account until the rules are accepted"
            )
    except Exception as exc:  # noqa: BLE001
        meta["_error"] = f"{type(exc).__name__}: {exc}"

    print("\n--- competition metadata (verbatim from API) ---")
    for key, value in meta.items():
        print(f"  {key:28s} {value}")
    if meta.get("user_has_entered") is False:
        print("\n  >>> user_has_entered is False: accept the rules on the competition")
        print("  >>> page before the data listing below will work.")

    # ---- file listing ------------------------------------------------------ #
    # A competition with hundreds of thousands of files needs thousands of pages,
    # and Kaggle rate-limits well before the end. Two consequences shape this
    # loop: back off and retry rather than dying, and checkpoint every page so a
    # run that is interrupted resumes instead of starting over.
    import csv

    files_csv = out_dir / "competition_files.csv"
    state_path = out_dir / "listing_state.json"

    rows: list[tuple[str, int, str]] = []
    token = None
    pages = 0

    if args.resume and files_csv.exists() and state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("competition") == args.competition:
            with files_csv.open(newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                next(reader, None)
                rows = [(r[0], int(r[1]), r[2]) for r in reader if len(r) >= 3]
            token = state.get("next_page_token") or None
            pages = int(state.get("pages", 0))
            if token:
                print(f"\n  resuming after page {pages} with {len(rows):,} files already listed")
            else:
                print(f"\n  previous listing was complete: {len(rows):,} files")

    def write_checkpoint(next_token, page_count):
        with files_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["name", "total_bytes", "creation_date"])
            writer.writerows(rows)
        state_path.write_text(
            json.dumps(
                {
                    "competition": args.competition,
                    "next_page_token": next_token,
                    "pages": page_count,
                    "n_files": len(rows),
                    "updated": datetime.now(UTC).isoformat(timespec="seconds"),
                },
                indent=2,
            )
        )

    def fetch_page(page_token):
        """One page, with backoff on rate limits and transient server errors."""
        delay = args.backoff
        for attempt in range(args.max_retries + 1):
            try:
                return api.competition_list_files(
                    args.competition, page_token=page_token, page_size=args.page_size
                )
            except Exception as exc:  # noqa: BLE001
                status = getattr(getattr(exc, "response", None), "status_code", None)
                retryable = status in (429, 500, 502, 503, 504)
                if not retryable or attempt == args.max_retries:
                    raise
                wait = delay
                retry_after = getattr(getattr(exc, "response", None), "headers", {}) or {}
                if retry_after.get("Retry-After", "").isdigit():
                    wait = max(wait, int(retry_after["Retry-After"]))
                print(f"\n  HTTP {status}; waiting {wait:.0f}s "
                      f"(retry {attempt + 1}/{args.max_retries})")
                time.sleep(wait)
                delay = min(delay * 2, args.max_backoff)
        raise RuntimeError("unreachable")

    print("\n--- listing files (names + sizes only, nothing downloaded) ---")
    interrupted = False
    complete = bool(rows) and token is None and pages > 0
    try:
        while not complete:
            page = fetch_page(token)
            files = list(getattr(page, "files", None) or [])
            for f in files:
                created = getattr(f, "creation_date", None)
                rows.append(
                    (
                        f.name,
                        int(getattr(f, "total_bytes", 0) or 0),
                        created.isoformat() if isinstance(created, datetime) else str(created),
                    )
                )
            pages += 1
            token = getattr(page, "next_page_token", None) or None
            if pages % 25 == 0 or not token:
                write_checkpoint(token, pages)
            print(f"  page {pages:>5d}  files so far: {len(rows):,}", end="\r", flush=True)
            if not token or not files:
                complete = True
                break
            if args.max_pages and pages >= args.max_pages:
                print(f"\n  stopped at --max-pages={args.max_pages}; listing is PARTIAL")
                break
            if args.sleep:
                time.sleep(args.sleep)
    except KeyboardInterrupt:
        interrupted = True
        print("\n  interrupted; checkpoint saved, re-run to resume")
    except Exception as exc:  # noqa: BLE001
        status = getattr(getattr(exc, "response", None), "status_code", None)
        print(f"\n  LISTING STOPPED: {type(exc).__name__}: {exc}")
        if status == 403:
            print("  403 means the competition rules have not been accepted.")
        elif status == 429:
            print("  429 is rate limiting. The checkpoint is saved — re-run to resume,")
            print("  optionally with --sleep 1 to pace the requests.")
        interrupted = True

    write_checkpoint(token, pages)
    partial = bool(token) or interrupted
    print(f"\n  pages fetched: {pages}   files listed: {len(rows):,}"
          f"{'  (PARTIAL — re-run to resume)' if partial else '  (COMPLETE)'}")

    if not rows:
        print("no files listed; nothing to summarise")
        return 3


    (out_dir / "competition_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )

    # ---- summarise --------------------------------------------------------- #
    total_bytes = sum(r[1] for r in rows)
    by_dir_bytes: dict[str, int] = defaultdict(int)
    by_dir_count: Counter[str] = Counter()
    by_ext_bytes: dict[str, int] = defaultdict(int)
    by_ext_count: Counter[str] = Counter()
    for name, size, _ in rows:
        by_dir_bytes[top_level(name)] += size
        by_dir_count[top_level(name)] += 1
        by_ext_bytes[extension(name)] += size
        by_ext_count[extension(name)] += 1

    def table(title: str, counts: Counter[str], sizes: dict[str, int]) -> list[str]:
        lines = [f"| {title} | files | bytes | size |", "|---|---:|---:|---:|"]
        for key, count in counts.most_common():
            lines.append(f"| `{key}` | {count:,} | {sizes[key]:,} | {human_bytes(sizes[key])} |")
        return lines

    small_files = sorted(
        (r for r in rows if r[1] < 200 * 1024 * 1024 and extension(r[0]) in {".csv", ".json", ".txt", ".parquet"}),
        key=lambda r: r[0],
    )

    def build_md(public_safe: bool) -> str:
        """Render the summary.

        public_safe=True omits every nested file path. File paths under the
        image directories embed StudyInstanceUIDs, so only aggregates and
        root-level file names may leave a private context (competition data
        may not be redistributed outside the team).
        """
        md: list[str] = [
            f"# Phase 0 step 1 — file inventory for `{args.competition}`",
            "",
            f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} "
            f"by `eda/phase0_01_auth_and_files.py`. Nothing was downloaded.",
            "",
        ]
        if public_safe:
            md += [
                "> Redacted view: aggregates and root-level file names only. "
                "Nested paths embed StudyInstanceUIDs and stay out of this file.",
                "",
            ]
        md += [
            "## Competition metadata (verbatim from the Kaggle API)",
            "",
            "| field | value |",
            "|---|---|",
        ]
        md += [f"| `{k}` | {v} |" for k, v in meta.items()]
        md += [
            "",
            "## Totals",
            "",
            f"- files listed: **{len(rows):,}**"
            f"{'  ⚠️ PARTIAL — the listing did not finish' if partial else ''}",
            f"- total size: **{human_bytes(total_bytes)}** ({total_bytes:,} bytes)",
            "",
            "## By top-level directory",
            "",
        ]
        md += table("directory", by_dir_count, by_dir_bytes)
        md += ["", "## By extension", ""]
        md += table("extension", by_ext_count, by_ext_bytes)
        md += [
            "",
            "## Small non-image files (the only ones worth downloading locally)",
            "",
            "| file | size |",
            "|---|---:|",
        ]
        shown = [r for r in small_files if not (public_safe and "/" in r[0])]
        md += [f"| `{n}` | {human_bytes(s)} |" for n, s, _ in shown[:100]]
        if len(shown) > 100:
            md.append(f"| … {len(shown) - 100:,} more | |")
        hidden = len(small_files) - len(shown)
        if hidden:
            md.append(f"| … {hidden:,} nested paths redacted | |")
        md.append("")
        return "\n".join(md)

    summary_path = out_dir / "step1_summary.md"
    summary_path.write_text(build_md(public_safe=False), encoding="utf-8")
    public_path = out_dir / "step1_summary_public.md"
    public_path.write_text(build_md(public_safe=True), encoding="utf-8")

    # ---- console report ---------------------------------------------------- #
    print(f"\n  total size: {human_bytes(total_bytes)} ({total_bytes:,} bytes)")
    print("\n  by top-level directory:")
    for key, count in by_dir_count.most_common(20):
        print(f"    {key:40s} {count:>10,} files  {human_bytes(by_dir_bytes[key]):>12s}")
    print("\n  by extension:")
    for key, count in by_ext_count.most_common(20):
        print(f"    {key:40s} {count:>10,} files  {human_bytes(by_ext_bytes[key]):>12s}")
    print(f"\n  small non-image files ({len(small_files)}):")
    for name, size, _ in small_files[:40]:
        print(f"    {name:60s} {human_bytes(size):>12s}")

    print(f"\nwrote {files_csv}")
    print(f"wrote {(out_dir / 'competition_meta.json')}")
    print(f"wrote {summary_path}  <- merge into docs/FINDINGS.md")
    print(f"wrote {public_path}  <- redacted, safe to paste anywhere")
    return 0


if __name__ == "__main__":
    sys.exit(main())
