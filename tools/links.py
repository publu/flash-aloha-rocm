#!/usr/bin/env python3
"""Extract outside links from text files of an untrusted checkout (read-only).

Usage: python3 -I links.py <repo-root> [--docs-only] [--skip-lock]

Prints "<url>\t<relpath>:<line>" for every http(s)/git/ssh URL and for bare
"something.github.io/..." style references. Nothing is imported or executed
from the repository.
"""
import os
import re
import sys

SKIP_DIRS = {".git"}
URL_RE = re.compile(r"""(?:(?:https?|git|ssh|ftp|wss?)://|git@)[^\s<>"'`\)\]\}\\,;]+""")
BARE_RE = re.compile(r"""(?<![/\w@.-])((?:[\w-]+\.)+(?:github\.io|com|org|io|ai|dev|net|edu|cn)/[^\s<>"'`\)\]\}\\,;]+)""")
DOC_EXT = {".md", ".rst", ".txt"}


def is_text(path):
    try:
        with open(path, "rb") as fh:
            head = fh.read(8192)
    except OSError:
        return False
    return b"\x00" not in head


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    root = os.path.abspath(argv[1])
    docs_only = "--docs-only" in argv
    skip_lock = "--skip-lock" in argv
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            ext = os.path.splitext(name)[1].lower()
            if docs_only and ext not in DOC_EXT and name not in ("LICENSE",):
                continue
            if skip_lock and name.endswith(".lock"):
                continue
            if ext in {".urdf", ".obj", ".stl"}:
                # huge model files; still scanned unless docs-only, but cheaply
                pass
            if not is_text(full):
                continue
            try:
                with open(full, "r", encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, 1):
                        if "://" not in line and "git@" not in line and "." not in line:
                            continue
                        seen = set()
                        for m in URL_RE.finditer(line):
                            url = m.group(0).rstrip(".:")
                            seen.add(url)
                            print(f"{url}\t{rel}:{lineno}")
                        if docs_only:
                            for m in BARE_RE.finditer(line):
                                bare = m.group(1).rstrip(".:")
                                if any(bare in s for s in seen):
                                    continue
                                print(f"(bare) {bare}\t{rel}:{lineno}")
            except OSError:
                continue
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
