#!/usr/bin/env python3
"""List third-party imports of an untrusted checkout by PARSING (ast), never importing.

Usage: python3 -I imports.py <repo-root> [--where MODULE]

Collects every `import X` / `from X import ...` (including those nested in
functions and try-blocks) from all .py files, drops the standard library and
modules that are defined inside the checkout, and prints each remaining
top-level module with how many files use it and where it is first used.
With --where MODULE, prints every file:line importing that module.
Also prints `if __name__ == "__main__"` entry points and argparse usage counts
with --mains.
"""
import ast
import os
import sys

SKIP_DIRS = {".git"}


def py_files(root):
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            if name.endswith(".py"):
                full = os.path.join(dirpath, name)
                yield full, os.path.relpath(full, root)


def local_modules(root):
    """Names importable from the checkout's own source roots."""
    names = set()
    source_roots = ["src", "scripts", "scripts/tests", "src/hardware/arx5-sdk/python",
                    "src/hardware/arx5-sdk/python/examples", "src/hardware/arx5-sdk/tests",
                    "src/kdl_parser/kdl_parser_py", "scripts/setup", "scripts/cameras",
                    "src/camera_utils", "src/config", "src/policy", "src/inference",
                    "src/recording", "src/visualization", "src/utils", "src/hardware",
                    "src/data_processing", "src/contracts", "src/visualization/visualization_utils",
                    "src/hardware/arx5-sdk/python/peripherals"]
    for rel in source_roots:
        base = os.path.join(root, rel)
        if not os.path.isdir(base):
            continue
        for entry in os.listdir(base):
            full = os.path.join(base, entry)
            if entry.endswith(".py"):
                names.add(entry[:-3])
            elif os.path.isdir(full) and any(f.endswith(".py") for f in os.listdir(full)):
                names.add(entry)
    return names


def is_main_guard(node):
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.comparators) != 1:
        return False
    left, right = test.left, test.comparators[0]
    pair = []
    for side in (left, right):
        if isinstance(side, ast.Name):
            pair.append(("name", side.id))
        elif isinstance(side, ast.Constant):
            pair.append(("const", side.value))
    return ("name", "__name__") in pair and ("const", "__main__") in pair


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    root = os.path.abspath(argv[1])
    where = argv[argv.index("--where") + 1] if "--where" in argv else None
    show_mains = "--mains" in argv
    stdlib = set(sys.stdlib_module_names)
    local = local_modules(root)
    uses = {}
    mains = []
    errors = []
    for full, rel in py_files(root):
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                source = fh.read()
            tree = ast.parse(source, filename=rel)
        except SyntaxError as exc:
            errors.append(f"{rel}: {exc.msg} (line {exc.lineno})")
            continue
        has_main = any(is_main_guard(node) for node in tree.body)
        uses_argparse = False
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [(alias.name, node.lineno) for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    mods = [(node.module, node.lineno)]
            for mod, lineno in mods:
                top = mod.split(".")[0]
                if top == "argparse":
                    uses_argparse = True
                uses.setdefault(top, []).append((rel, lineno, mod))
        if has_main:
            mains.append((rel, uses_argparse, source.count("\n") + 1))
    if where:
        for rel, lineno, mod in uses.get(where, []):
            print(f"{rel}:{lineno}\t{mod}")
        return 0
    if show_mains:
        print("files with an `if __name__ == '__main__'` guard (path, lines, argparse?):")
        for rel, uses_argparse, lines in mains:
            print(f"  {rel}\t{lines}\t{'argparse' if uses_argparse else '-'}")
        print(f"total: {len(mains)}")
        return 0
    print(f"{'module':28s} {'files':>5s}  kind         first use")
    for top in sorted(uses, key=lambda m: (m in stdlib, m in local, m.lower())):
        if top in stdlib:
            continue
        kind = "local" if top in local else "third-party"
        files = sorted({rel for rel, _l, _m in uses[top]})
        rel, lineno, mod = uses[top][0]
        print(f"{top:28s} {len(files):5d}  {kind:12s} {rel}:{lineno} ({mod})")
    if errors:
        print("---- files that did not parse ----")
        for line in errors:
            print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
