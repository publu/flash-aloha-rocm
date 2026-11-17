#!/usr/bin/env python3
"""Check relative Markdown links (and #anchors) of an untrusted checkout, read-only.

Usage: python3 -I mdlinks.py <repo-root> [--all]

For every [text](target) in .md files whose target is not an absolute URL,
resolves it against the file's directory and reports targets that do not exist
and anchors that match no heading (GitHub-style slug) of the target file.
Also reports back-ticked repository paths in the top-level README that do not
exist (a cheap check of what the README points at). With --all, prints every
checked link.
"""
import os
import re
import sys

SKIP_DIRS = {".git"}
LINK_RE = re.compile(r"(?<!\!)\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
CODE_PATH_RE = re.compile(r"`([A-Za-z0-9_.\-/]+\.(?:py|sh|md|yaml|yml|npz|npy|json|toml|urdf|cpp|h))`")


def slug(heading):
    text = heading.strip().lower()
    text = re.sub(r"`", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
    return text.replace(" ", "-")


def anchors_of(path):
    found = set()
    counts = {}
    in_code = False
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.lstrip().startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue
                match = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
                if match:
                    base = slug(match.group(2))
                    n = counts.get(base, 0)
                    counts[base] = n + 1
                    found.add(base if n == 0 else f"{base}-{n}")
    except OSError:
        pass
    return found


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    root = os.path.abspath(argv[1])
    show_all = "--all" in argv
    checked = broken = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            if not name.lower().endswith(".md"):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            in_code = False
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                for lineno, line in enumerate(fh, 1):
                    if line.lstrip().startswith("```"):
                        in_code = not in_code
                        continue
                    if in_code:
                        continue
                    for match in LINK_RE.finditer(line):
                        target = match.group(2)
                        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:", target):
                            continue  # absolute URL / mailto
                        checked += 1
                        path_part, _, anchor = target.partition("#")
                        dest = full if not path_part else os.path.normpath(os.path.join(dirpath, path_part))
                        problem = None
                        if not os.path.exists(dest):
                            problem = "missing file"
                        elif not os.path.abspath(dest).startswith(root):
                            problem = "points outside the repository"
                        elif anchor and os.path.isfile(dest) and dest.lower().endswith(".md"):
                            if anchor.lower() not in anchors_of(dest):
                                problem = f"no heading for #{anchor}"
                        if problem:
                            broken += 1
                            print(f"BROKEN  {rel}:{lineno}  -> {target}  ({problem})")
                        elif show_all:
                            print(f"ok      {rel}:{lineno}  -> {target}")
    print(f"relative markdown links checked: {checked}, broken: {broken}")
    readme = os.path.join(root, "README.md")
    missing = []
    total = 0
    if os.path.isfile(readme):
        with open(readme, "r", encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, 1):
                for match in CODE_PATH_RE.finditer(line):
                    candidate = match.group(1)
                    if "/" not in candidate:
                        continue
                    total += 1
                    if not os.path.exists(os.path.join(root, candidate)):
                        missing.append((lineno, candidate))
    print(f"back-ticked repository paths in README.md: {total}, not present in the checkout: {len(missing)}")
    for lineno, candidate in missing:
        print(f"  README.md:{lineno}  `{candidate}`")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
