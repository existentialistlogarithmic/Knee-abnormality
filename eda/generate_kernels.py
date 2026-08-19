#!/usr/bin/env python3
"""Render every Kaggle kernel from `src/pipeline.py` plus `kaggle/_templates/`.

    python eda/generate_kernels.py            # show what would change
    python eda/generate_kernels.py --write    # write it
    python eda/generate_kernels.py --check    # exit 1 if the tree has drifted

Kaggle script kernels are single files, so sharing code between them means
splicing it in at generation time. A template is an ordinary Python file with
two kinds of placeholder:

    @@CONFIG@@          the constants this kernel runs with, from the manifest
    @@INCLUDE name@@    the body of kaggle/_templates/_shared/name.py

`--check` is the part that matters. It is what stops the manifest from becoming
documentation of a tree that has since been edited by hand — the failure mode
this whole exercise exists to remove.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import Kernel, Raw, all_kernels, check  # noqa: E402

TEMPLATES = ROOT / "kaggle" / "_templates"
BANNER = "# " + "-" * 75 + " #"


def literal(value: object) -> str:
    if isinstance(value, Raw):
        return str(value)
    if isinstance(value, str):
        return '"' + value.replace('"', '\\"') + '"'
    return repr(value)


def render_config(kernel: Kernel) -> str:
    """The constants block, plus whatever rationale the manifest carries."""
    lines = []
    if kernel.note:
        lines += [f"# {line}" if line else "#" for line in kernel.note.split("\n")]
        lines.append("#")
    width = max(len(name) for name in kernel.constants)
    for name, value in kernel.constants.items():
        lines.append(f"{name.ljust(width)} = {literal(value)}")
    return "\n".join(lines)


def shared_body(name: str, wanted: list[str] | None = None) -> str:
    """A shared module's source, minus its docstring.

    `wanted` selects top-level definitions by name. Kernels differ in which
    helpers they need — a CPU cache build has no use for the GPU check — and
    splicing in unused code would put a torch import in a kernel that has no
    GPU. Selecting keeps each generated file to what it actually runs.
    """
    source = (TEMPLATES / "_shared" / f"{name}.py").read_text()
    body = source.split('"""', 2)[2].strip("\n")
    if not wanted:
        return body
    lines = body.splitlines(keepends=True)
    tree = ast.parse(body)
    keep, unknown = [], set(wanted)
    for node in tree.body:
        node_name = getattr(node, "name", None)
        if node_name in wanted:
            unknown.discard(node_name)
            keep.append("".join(lines[node.lineno - 1:node.end_lineno]).rstrip("\n"))
        elif node_name is None:
            keep.append("".join(lines[node.lineno - 1:node.end_lineno]).rstrip("\n"))
    if unknown:
        raise SystemExit(f"{name}.py has no {sorted(unknown)}")
    return "\n\n\n".join(keep)


def render(kernel: Kernel) -> str:
    template = (TEMPLATES / f"{kernel.template}.py.in").read_text()
    out = []
    for line in template.splitlines():
        if line.strip() == "@@CONFIG@@":
            out.append(render_config(kernel))
        elif line.strip().startswith("@@INCLUDE "):
            spec = line.strip()[len("@@INCLUDE "):-2].strip()
            name, _, wanted = spec.partition(":")
            body = shared_body(name, wanted.split(",") if wanted else None)
            out.append(f"\n{BANNER}\n# from kaggle/_templates/_shared/{name}.py\n"
                       f"{BANNER}\n{body}\n")
        else:
            out.append(line)
    text = "\n".join(out)
    while "\n\n\n\n" in text:
        text = text.replace("\n\n\n\n", "\n\n\n")
    return text.rstrip() + "\n"


def outputs(kernel: Kernel) -> dict[Path, str]:
    directory = ROOT / "kaggle" / kernel.directory
    return {
        directory / "run.py": render(kernel),
        directory / "kernel-metadata.json":
            json.dumps(kernel.metadata(), indent=2) + "\n",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write the files")
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if the tree differs from the manifest")
    parser.add_argument("--diff", action="store_true", help="show full diffs")
    args = parser.parse_args()

    problems = check()
    for problem in problems:
        print(f"MANIFEST PROBLEM: {problem}")
    if problems:
        return 2

    changed, written = [], 0
    for kernel in all_kernels():
        for path, wanted in outputs(kernel).items():
            current = path.read_text() if path.exists() else None
            if current == wanted:
                continue
            changed.append(path.relative_to(ROOT))
            if args.diff:
                print("".join(difflib.unified_diff(
                    (current or "").splitlines(keepends=True),
                    wanted.splitlines(keepends=True),
                    f"a/{path.relative_to(ROOT)}", f"b/{path.relative_to(ROOT)}")))
            if args.write:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(wanted)
                written += 1

    kernels = all_kernels()
    print(f"{len(kernels)} kernels, {len(kernels) * 2} files")
    if not changed:
        print("tree matches the manifest")
        return 0

    print(f"{len(changed)} file(s) differ:")
    for path in changed:
        print(f"  {path}")
    if args.write:
        print(f"wrote {written}")
        return 0
    if args.check:
        print("run: python eda/generate_kernels.py --write")
        return 1
    print("(dry run — pass --write to apply, --diff to see the changes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
