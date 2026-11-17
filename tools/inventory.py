#!/usr/bin/env python3
"""Read-only inventory of an untrusted repository checkout.

Usage: python3 -I inventory.py <repo-root> <command> [args]

Commands:
  tree [depth]     directory tree limited to depth (default 3), with file counts
  ext              file count and bytes by extension
  largest [n]      n largest files (default 40)
  lfs              files that are git-lfs pointers (start with 'version https://git-lfs')
  binaries         files that look binary (NUL byte in first 8 KiB) + magic sniff
  exec             files with an executable bit, with their shebang

Nothing in the repository is imported or executed; files are only opened for reading.
"""
import os
import stat
import sys
from collections import defaultdict

SKIP_DIRS = {".git"}


def walk(root):
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            yield full, os.path.relpath(full, root)


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0


def cmd_tree(root, depth=3):
    depth = int(depth)

    def count(path):
        files = 0
        size = 0
        for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                files += 1
                try:
                    size += os.lstat(os.path.join(dirpath, name)).st_size
                except OSError:
                    pass
        return files, size

    def rec(path, level, prefix):
        try:
            entries = sorted(os.listdir(path))
        except OSError as exc:
            print(prefix + f"[unreadable: {exc}]")
            return
        dirs = [e for e in entries if os.path.isdir(os.path.join(path, e)) and not os.path.islink(os.path.join(path, e)) and e not in SKIP_DIRS]
        files = [e for e in entries if e not in dirs and e not in SKIP_DIRS]
        for d in dirs:
            full = os.path.join(path, d)
            nfiles, size = count(full)
            print(f"{prefix}{d}/  ({nfiles} files, {human(size)})")
            if level < depth:
                rec(full, level + 1, prefix + "    ")
        if level <= depth:
            # Collapse long file lists so the tree stays readable.
            limit = 60 if level == 1 else 25
            shown = files[:limit]
            for f in shown:
                full = os.path.join(path, f)
                try:
                    size = os.lstat(full).st_size
                except OSError:
                    size = -1
                link = " -> " + os.readlink(full) if os.path.islink(full) else ""
                print(f"{prefix}{f}  [{human(size)}]{link}")
            if len(files) > limit:
                print(f"{prefix}... (+{len(files) - limit} more files)")

    nfiles, size = count(root)
    print(f"./  ({nfiles} files, {human(size)})")
    rec(root, 1, "    ")


def cmd_ext(root):
    counts = defaultdict(int)
    sizes = defaultdict(int)
    for full, rel in walk(root):
        base = os.path.basename(rel)
        if base.startswith(".") and base.count(".") == 1:
            ext = base  # dotfile such as .gitignore
        else:
            ext = os.path.splitext(base)[1].lower() or "(none:" + base + ")"
        counts[ext] += 1
        sizes[ext] += os.lstat(full).st_size
    total_n = sum(counts.values())
    total_s = sum(sizes.values())
    print(f"{'extension':28s} {'files':>6s} {'bytes':>12s}  share-of-bytes")
    for ext in sorted(counts, key=lambda e: (-counts[e], e)):
        print(f"{ext:28s} {counts[ext]:6d} {sizes[ext]:12d}  {100.0 * sizes[ext] / max(total_s, 1):5.1f}%")
    print(f"{'TOTAL':28s} {total_n:6d} {total_s:12d}")


def cmd_largest(root, n=40):
    n = int(n)
    rows = []
    for full, rel in walk(root):
        rows.append((os.lstat(full).st_size, rel))
    rows.sort(reverse=True)
    for size, rel in rows[:n]:
        print(f"{size:12d}  {human(size):>10s}  {rel}")


def cmd_lfs(root):
    found = 0
    for full, rel in walk(root):
        try:
            with open(full, "rb") as fh:
                head = fh.read(200)
        except OSError:
            continue
        if head.startswith(b"version https://git-lfs"):
            found += 1
            print(rel)
            print("    " + head.decode("utf-8", "replace").replace("\n", " | "))
    print(f"lfs pointer files: {found}")


MAGIC = [
    (b"\x7fELF", "ELF executable/shared object"),
    (b"MZ", "PE/DOS executable"),
    (b"\xcf\xfa\xed\xfe", "Mach-O 64"),
    (b"\xfe\xed\xfa\xce", "Mach-O 32"),
    (b"\xca\xfe\xba\xbe", "Mach-O fat / Java class"),
    (b"PK\x03\x04", "zip container (zip/whl/jar/npz/docx/pt)"),
    (b"\x1f\x8b", "gzip"),
    (b"BZh", "bzip2"),
    (b"\xfd7zXZ\x00", "xz"),
    (b"7z\xbc\xaf\x27\x1c", "7z"),
    (b"\x89PNG", "PNG image"),
    (b"\xff\xd8\xff", "JPEG image"),
    (b"GIF8", "GIF image"),
    (b"%PDF", "PDF"),
    (b"\x89HDF\r\n\x1a\n", "HDF5"),
    (b"\x93NUMPY", "NumPy .npy"),
    (b"\x80\x02", "pickle protocol 2"),
    (b"\x80\x03", "pickle protocol 3"),
    (b"\x80\x04", "pickle protocol 4"),
    (b"\x80\x05", "pickle protocol 5"),
    (b"RIFF", "RIFF (wav/avi/webp)"),
    (b"\x00\x00\x00\x18ftyp", "MP4"),
    (b"\x00\x00\x00\x1cftyp", "MP4"),
    (b"\x00\x00\x00 ftyp", "MP4"),
    (b"solid", "STL ascii (maybe)"),
    (b"!<arch>", "ar archive / static lib"),
    (b"SQLite format 3", "SQLite db"),
    (b"glTF", "glTF binary"),
]


def sniff(head):
    for magic, label in MAGIC:
        if head.startswith(magic):
            return label
    return "unknown binary"


def cmd_binaries(root):
    n = 0
    for full, rel in walk(root):
        try:
            with open(full, "rb") as fh:
                head = fh.read(8192)
        except OSError:
            continue
        if b"\x00" in head:
            n += 1
            size = os.lstat(full).st_size
            print(f"{size:10d}  {sniff(head):40s}  {rel}")
    print(f"binary-looking files: {n}")


def cmd_exec(root):
    n = 0
    for full, rel in walk(root):
        mode = os.lstat(full).st_mode
        if mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
            n += 1
            try:
                with open(full, "rb") as fh:
                    first = fh.readline(200)
            except OSError:
                first = b""
            line = first.decode("utf-8", "replace").rstrip("\n") if first.startswith(b"#!") else "(no shebang)"
            print(f"{rel}  ::  {line}")
    print(f"executable-bit files: {n}")


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    root = os.path.abspath(argv[1])
    cmd = argv[2]
    args = argv[3:]
    table = {
        "tree": cmd_tree,
        "ext": cmd_ext,
        "largest": cmd_largest,
        "lfs": cmd_lfs,
        "binaries": cmd_binaries,
        "exec": cmd_exec,
    }
    if cmd not in table:
        print(__doc__)
        return 2
    table[cmd](root, *args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
