#!/usr/bin/env python3
"""Static code map of an untrusted checkout. PARSES files with ast; never imports them.

Usage (always with -I, from the tools directory):
  python3 -I codemap.py outline  <repo-root> [--prefix PATH] [--vendored] [--private] [--methods]
  python3 -I codemap.py argparse <repo-root> [--prefix PATH] [--vendored]
  python3 -I codemap.py imports  <repo-root> [--prefix PATH] [--vendored]
  python3 -I codemap.py localdeps <repo-root> [--prefix PATH]
  python3 -I codemap.py markers  <repo-root> [--vendored]
  python3 -I codemap.py calls    <repo-root> NAME[,NAME...] [--prefix PATH] [--vendored]
  python3 -I codemap.py env      <repo-root> [--vendored]
  python3 -I codemap.py tests    <repo-root>
  python3 -I codemap.py unused   <repo-root>

outline   : per file: line count, module docstring (first line), classes / functions with line numbers.
argparse  : every parser.add_argument(...) call with its flags, type, default, choices, action, required.
imports   : per file third-party / local / stdlib imports.
localdeps : package-to-package import matrix for the checkout's own packages.
markers   : TODO / FIXME / XXX / HACK / NOTE counts and lines.
calls     : file:line of every call whose dotted name ends with one of NAME.
env       : os.environ / os.getenv reads with the variable name.
tests     : per test file: number of test functions, skip / skipif / importorskip / parametrize markers,
            and which own modules it imports.
unused    : top-level functions/classes defined in own non-test modules whose name appears in no other
            .py file of the checkout (candidates for dead code; names used only in their own file are listed
            separately).
"""
import ast
import os
import re
import sys

VENDORED = ("src/hardware/arx5-sdk/", "src/kdl_parser/")
SKIP_DIRS = {".git"}


def iter_py(root, prefix=None, vendored=False):
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            if not vendored and rel.startswith(VENDORED):
                continue
            if prefix and not rel.startswith(prefix):
                continue
            yield full, rel


def parse(full):
    with open(full, "rb") as handle:
        data = handle.read()
    text = data.decode("utf-8", errors="replace")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:  # report, do not crash
        return text, None, exc
    return text, tree, None


def first_line(doc):
    if not doc:
        return ""
    line = doc.strip().splitlines()[0].strip()
    return line if len(line) <= 150 else line[:147] + "..."


def dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    elif isinstance(node, ast.Call):
        parts.append(dotted(node.func) + "()")
    else:
        parts.append("?")
    return ".".join(reversed(parts))


def sig(fn):
    args = fn.args
    names = [a.arg for a in args.posonlyargs + args.args]
    if args.vararg:
        names.append("*" + args.vararg.arg)
    elif args.kwonlyargs:
        names.append("*")
    names += [a.arg for a in args.kwonlyargs]
    if args.kwarg:
        names.append("**" + args.kwarg.arg)
    text = ", ".join(names)
    return text if len(text) <= 110 else text[:107] + "..."


def cmd_outline(root, prefix, vendored, private, methods):
    for full, rel in iter_py(root, prefix, vendored):
        text, tree, err = parse(full)
        nlines = text.count("\n") + (0 if text.endswith("\n") or not text else 1)
        if tree is None:
            print(f"## {rel}  [{nlines} lines]  SYNTAX ERROR: {err}")
            continue
        has_main = any(_is_main_guard(n) for n in tree.body)
        print(f"## {rel}  [{nlines} lines]{'  __main__' if has_main else ''}")
        doc = first_line(ast.get_docstring(tree))
        if doc:
            print(f"   doc: {doc}")
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                bases = ", ".join(dotted(b) for b in node.bases)
                end = getattr(node, "end_lineno", node.lineno)
                print(f"   class {node.name}({bases})  L{node.lineno}-{end}  {first_line(ast.get_docstring(node))}")
                if methods:
                    for sub in node.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if not private and sub.name.startswith("_") and sub.name not in ("__init__", "__call__"):
                                continue
                            send = getattr(sub, "end_lineno", sub.lineno)
                            kind = "async def" if isinstance(sub, ast.AsyncFunctionDef) else "def"
                            print(f"      {kind} {sub.name}({sig(sub)})  L{sub.lineno}-{send}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not private and node.name.startswith("_"):
                    continue
                end = getattr(node, "end_lineno", node.lineno)
                kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                print(f"   {kind} {node.name}({sig(node)})  L{node.lineno}-{end}  {first_line(ast.get_docstring(node))}")
            elif isinstance(node, ast.Assign) and not methods:
                continue
        if methods:
            consts = []
            for node in tree.body:
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            consts.append(f"{target.id}@L{node.lineno}")
            if consts:
                joined = ", ".join(consts)
                print(f"   consts: {joined if len(joined) < 900 else joined[:900] + ' ...'}")


def _is_main_guard(node):
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.comparators) != 1:
        return False
    values = []
    for side in (test.left, test.comparators[0]):
        if isinstance(side, ast.Name):
            values.append(side.id)
        elif isinstance(side, ast.Constant):
            values.append(side.value)
    return "__name__" in values and "__main__" in values


def literal(node):
    if node is None:
        return None
    try:
        return repr(ast.literal_eval(node))
    except Exception:
        try:
            return ast.unparse(node)
        except Exception:
            return "<expr>"


def cmd_argparse(root, prefix, vendored):
    for full, rel in iter_py(root, prefix, vendored):
        _text, tree, _err = parse(full)
        if tree is None:
            continue
        rows = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = dotted(node.func)
            if name.endswith(".add_argument"):
                flags = [literal(a) for a in node.args]
                kw = {k.arg: literal(k.value) for k in node.keywords if k.arg}
                rows.append((node.lineno, "arg", flags, kw, name))
            elif name.endswith(".add_parser") or name.endswith(".add_subparsers") or name.endswith("ArgumentParser") \
                    or name.endswith(".add_mutually_exclusive_group") or name.endswith(".set_defaults"):
                flags = [literal(a) for a in node.args]
                kw = {k.arg: literal(k.value) for k in node.keywords if k.arg}
                rows.append((node.lineno, name.rsplit(".", 1)[-1], flags, kw, name))
        if not rows:
            continue
        print(f"## {rel}")
        for lineno, kind, flags, kw, name in sorted(rows, key=lambda r: r[0]):
            keep = {k: v for k, v in kw.items() if k in ("type", "default", "choices", "action", "required", "nargs", "dest", "const")}
            helptext = kw.get("help")
            if helptext and len(helptext) > 140:
                helptext = helptext[:137] + "...'"
            extra = "  ".join(f"{k}={v}" for k, v in keep.items())
            if kind == "arg":
                print(f"   L{lineno}: {' '.join(flags)}  {extra}{'  help=' + helptext if helptext else ''}")
            else:
                descr = kw.get("description") or kw.get("help") or ""
                if len(descr) > 140:
                    descr = descr[:137] + "...'"
                print(f"   L{lineno}: [{kind}] {' '.join(flags)} {extra} {descr}".rstrip())


STDLIB = set(getattr(sys, "stdlib_module_names", ()))


def local_names(root):
    names = set()
    roots = ["src", "scripts", "scripts/tests", "scripts/setup", "scripts/cameras", "src/camera_utils", "src/config",
             "src/policy", "src/inference", "src/recording", "src/visualization", "src/utils", "src/hardware",
             "src/data_processing", "src/contracts", "src/visualization/visualization_utils",
             "src/hardware/arx5-sdk/python", "src/hardware/arx5-sdk/python/examples",
             "src/hardware/arx5-sdk/python/peripherals", "src/hardware/arx5-sdk/tests",
             "src/kdl_parser/kdl_parser_py"]
    for rel in roots:
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


def file_imports(tree):
    """Yield (lineno, module, names, nested) for every import in the tree."""
    top_nodes = set(id(n) for n in tree.body)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name, [], id(node) not in top_nodes
        elif isinstance(node, ast.ImportFrom):
            module = ("." * node.level) + (node.module or "")
            yield node.lineno, module, [a.name for a in node.names], id(node) not in top_nodes


def cmd_imports(root, prefix, vendored):
    local = local_names(root)
    for full, rel in iter_py(root, prefix, vendored):
        _text, tree, _err = parse(full)
        if tree is None:
            continue
        third, own, std = {}, {}, {}
        for lineno, module, names, nested in file_imports(tree):
            top = module.lstrip(".").split(".")[0]
            target = own if (module.startswith(".") or top in local) else std if top in STDLIB else third
            entry = target.setdefault(module, [lineno, nested])
            entry[1] = entry[1] and nested
        print(f"## {rel}")
        for label, bucket in (("third-party", third), ("own", own)):
            if bucket:
                items = ", ".join(f"{m}{'*' if v[1] else ''}@L{v[0]}" for m, v in sorted(bucket.items()))
                print(f"   {label}: {items}")


def cmd_localdeps(root, prefix):
    packages = ["camera_utils", "config", "contracts", "data_processing", "hardware", "inference", "policy",
                "recording", "utils", "visualization"]
    matrix = {}
    detail = {}
    for full, rel in iter_py(root, prefix, False):
        _text, tree, _err = parse(full)
        if tree is None:
            continue
        parts = rel.split("/")
        if parts[0] == "src" and len(parts) > 2:
            source = parts[1]
        elif parts[0] == "scripts":
            source = "scripts/tests" if len(parts) > 2 and parts[1] == "tests" else "scripts"
        else:
            source = parts[0]
        for _lineno, module, _names, _nested in file_imports(tree):
            top = module.lstrip(".").split(".")[0]
            if top in packages and top != source:
                matrix.setdefault(source, {}).setdefault(top, 0)
                matrix[source][top] += 1
                detail.setdefault((source, top), set()).add(module)
    for source in sorted(matrix):
        row = ", ".join(f"{t}({n})" for t, n in sorted(matrix[source].items()))
        print(f"{source} -> {row}")
    print()
    for (source, top), modules in sorted(detail.items()):
        print(f"  {source} -> {top}: {', '.join(sorted(modules))}")


MARKER = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")


def cmd_markers(root, vendored):
    counts = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            if not vendored and rel.startswith(VENDORED):
                continue
            if not name.endswith((".py", ".sh", ".md", ".yaml", ".yml", ".toml", ".cpp", ".h", ".hpp", ".txt", ".cfg")) \
                    and "." in name:
                continue
            try:
                with open(full, "rb") as handle:
                    data = handle.read()
            except OSError:
                continue
            if b"\0" in data[:4096]:
                continue
            for number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
                for match in MARKER.finditer(line):
                    counts.setdefault(match.group(1), []).append((rel, number, line.strip()[:160]))
    for marker in ("TODO", "FIXME", "XXX", "HACK"):
        rows = counts.get(marker, [])
        print(f"{marker}: {len(rows)}")
        for rel, number, line in rows:
            print(f"   {rel}:{number}: {line}")


def cmd_calls(root, names, prefix, vendored):
    wanted = [n.strip() for n in names.split(",") if n.strip()]
    for full, rel in iter_py(root, prefix, vendored):
        text, tree, _err = parse(full)
        if tree is None:
            continue
        lines = text.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = dotted(node.func)
                for want in wanted:
                    if name == want or name.endswith("." + want):
                        src = lines[node.lineno - 1].strip()[:170]
                        print(f"{rel}:{node.lineno}: [{name}] {src}")
                        break


def cmd_env(root, vendored):
    for full, rel in iter_py(root, None, vendored):
        text, tree, _err = parse(full)
        if tree is None:
            continue
        lines = text.splitlines()
        for node in ast.walk(tree):
            hit = None
            if isinstance(node, ast.Call):
                name = dotted(node.func)
                if name in ("os.getenv", "os.environ.get", "os.environ.setdefault", "os.environ.pop", "os.putenv",
                            "environ.get"):
                    hit = (name, literal(node.args[0]) if node.args else "?")
            elif isinstance(node, ast.Subscript) and dotted(node.value) in ("os.environ", "environ"):
                kind = "write" if isinstance(node.ctx, ast.Store) else "read"
                hit = (f"os.environ[] {kind}", literal(node.slice))
            if hit:
                print(f"{rel}:{node.lineno}: {hit[0]} {hit[1]}   | {lines[node.lineno - 1].strip()[:120]}")


def cmd_tests(root):
    local = local_names(root)
    total_tests = 0
    total_files = 0
    for full, rel in iter_py(root, "scripts/tests/", False):
        text, tree, _err = parse(full)
        if tree is None:
            continue
        tests = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                tests.append(node.name)
        marks = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = dotted(node.func)
                for key in ("pytest.skip", "pytest.importorskip", "pytest.mark.skipif", "pytest.mark.skip",
                            "pytest.mark.parametrize", "pytest.mark.xfail", "pytest.raises", "monkeypatch.setattr",
                            "subprocess.run", "subprocess.Popen", "pytest.fixture"):
                    if name == key or name.endswith("." + key):
                        marks[key] = marks.get(key, 0) + 1
        own = set()
        for _lineno, module, _names, _nested in file_imports(tree):
            top = module.lstrip(".").split(".")[0]
            if top in local and not top.startswith("test_"):
                own.add(module)
        if rel.rsplit("/", 1)[-1].startswith("test_"):
            total_files += 1
            total_tests += len(tests)
        nlines = text.count("\n")
        markstr = ", ".join(f"{k.replace('pytest.', '')}={v}" for k, v in sorted(marks.items()))
        print(f"## {rel}  [{nlines} lines, {len(tests)} tests]  {markstr}")
        print(f"   imports: {', '.join(sorted(own))}")
    print(f"\nTOTAL test files: {total_files}; test functions: {total_tests}")


def cmd_unused(root):
    texts = {}
    for full, rel in iter_py(root, None, False):
        text, tree, _err = parse(full)
        texts[rel] = (text, tree)
    # also count mentions in non-python own text files (docs, yaml, sh)
    other = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            if rel.startswith(VENDORED) or name.endswith(".py"):
                continue
            if name.endswith((".md", ".yaml", ".yml", ".sh", ".toml")):
                with open(os.path.join(dirpath, name), "rb") as handle:
                    other.append((rel, handle.read().decode("utf-8", errors="replace")))
    never, own_only_tests = [], []
    for rel, (text, tree) in sorted(texts.items()):
        if tree is None or rel.startswith("scripts/tests/"):
            continue
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            name = node.name
            if name in ("main",) or name.startswith("__"):
                continue
            pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])")
            own_count = len(pattern.findall(text))
            elsewhere_src, elsewhere_tests = [], []
            for other_rel, (other_text, _t) in texts.items():
                if other_rel == rel:
                    continue
                if pattern.search(other_text):
                    (elsewhere_tests if other_rel.startswith("scripts/tests/") else elsewhere_src).append(other_rel)
            in_docs = [r for r, t in other if pattern.search(t)]
            if own_count <= 1 and not elsewhere_src and not elsewhere_tests:
                never.append((rel, node.lineno, name, in_docs))
            elif own_count <= 1 and not elsewhere_src and elsewhere_tests:
                own_only_tests.append((rel, node.lineno, name, elsewhere_tests))
    print("Defined once and referenced nowhere else in any .py file (dead-code candidates):")
    for rel, lineno, name, in_docs in never:
        print(f"   {rel}:{lineno}: {name}{'  (mentioned in ' + ', '.join(in_docs[:3]) + ')' if in_docs else ''}")
    print(f"   -> {len(never)} names")
    print("\nReferenced only from tests (not from other source modules, not within own file):")
    for rel, lineno, name, tests in own_only_tests:
        print(f"   {rel}:{lineno}: {name}  <- {', '.join(t.rsplit('/', 1)[-1] for t in tests[:4])}")
    print(f"   -> {len(own_only_tests)} names")


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    command, root = argv[1], argv[2]
    rest = argv[3:]
    prefix = None
    if "--prefix" in rest:
        prefix = rest[rest.index("--prefix") + 1]
    vendored = "--vendored" in rest
    if command == "outline":
        cmd_outline(root, prefix, vendored, "--private" in rest, "--methods" in rest)
    elif command == "argparse":
        cmd_argparse(root, prefix, vendored)
    elif command == "imports":
        cmd_imports(root, prefix, vendored)
    elif command == "localdeps":
        cmd_localdeps(root, prefix)
    elif command == "markers":
        cmd_markers(root, vendored)
    elif command == "calls":
        cmd_calls(root, rest[0], prefix, vendored)
    elif command == "env":
        cmd_env(root, vendored)
    elif command == "tests":
        cmd_tests(root)
    elif command == "unused":
        cmd_unused(root)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
