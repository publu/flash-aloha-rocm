#!/usr/bin/env python3
"""Compare a vendored directory of an untrusted checkout with an upstream git tree.

Usage:
  python3 -I upstream_compare.py tree <repo-root> <vendored-prefix> <upstream-tree.json> [--list]
  python3 -I upstream_compare.py meta <github-repo.json> [<releases.json> <branches.json> <tags.json>]

`tree`: <upstream-tree.json> is the saved answer of the public GitHub API
"git/trees/<sha>?recursive=1". Local git blob ids are recomputed from the
work-tree bytes (sha1 of "blob <size>\\0<bytes>"), so nothing is executed and
git is not called. Prints identical / modified / only-local / only-upstream
files and local files whose content matches an upstream file at another path.

`meta`: prints selected public fields of the saved repository metadata.
Both inputs are treated as data only.
"""
import hashlib
import json
import os
import sys

SKIP_DIRS = {".git"}


def blob_sha(path):
    size = os.lstat(path).st_size
    digest = hashlib.sha1()
    digest.update(b"blob %d\x00" % size)
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest(), size


def cmd_tree(argv):
    root = os.path.abspath(argv[0])
    prefix = argv[1].strip("/")
    with open(argv[2], "r", encoding="utf-8") as fh:
        upstream = json.load(fh)
    show_list = "--list" in argv
    if upstream.get("truncated"):
        print("WARNING: upstream tree listing is truncated by the API")
    up = {}
    for entry in upstream.get("tree", []):
        if entry.get("type") == "blob":
            up[entry["path"]] = (entry["sha"], entry.get("size"), entry.get("mode"))
        elif entry.get("type") == "commit":
            up[entry["path"]] = ("(submodule) " + entry["sha"], None, entry.get("mode"))
    local = {}
    base = os.path.join(root, prefix)
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, base)
            sha, size = blob_sha(full)
            mode = "100755" if os.lstat(full).st_mode & 0o111 else "100644"
            local[rel] = (sha, size, mode)
    identical = sorted(p for p in local if p in up and local[p][0] == up[p][0])
    modified = sorted(p for p in local if p in up and local[p][0] != up[p][0])
    only_local = sorted(p for p in local if p not in up)
    only_up = sorted(p for p in up if p not in local)
    up_by_sha = {}
    for path, (sha, _size, _mode) in up.items():
        up_by_sha.setdefault(sha, []).append(path)
    print(f"upstream tree sha: {upstream.get('sha')}  entries: {len(upstream.get('tree', []))} "
          f"(blobs: {len(up)})")
    print(f"local files under {prefix}: {len(local)}")
    print(f"identical to upstream (same path, same blob id): {len(identical)} files, "
          f"{sum(local[p][1] for p in identical)} bytes")
    print(f"modified (same path, different content): {len(modified)} files")
    print(f"only in the local copy: {len(only_local)} files")
    print(f"only upstream (not carried locally): {len(only_up)} files")
    mode_diff = [p for p in identical if up[p][2] and up[p][2] != local[p][2]]
    if mode_diff:
        print(f"identical content but different file mode: {mode_diff}")
    if show_list:
        print("---- identical ----")
        for p in identical:
            print(f"  = {p}  [{local[p][1]} B]")
    print("---- modified ----")
    for p in modified:
        print(f"  M {p}  [local {local[p][1]} B, upstream {up[p][1]} B]")
    print("---- only local ----")
    for p in only_local:
        twin = up_by_sha.get(local[p][0])
        note = f"   (same content as upstream {twin})" if twin else ""
        print(f"  A {p}  [{local[p][1]} B]{note}")
    print("---- only upstream ----")
    for p in only_up:
        print(f"  D {p}  [{up[p][1]} B]")
    return 0


def cmd_meta(argv):
    with open(argv[0], "r", encoding="utf-8") as fh:
        repo = json.load(fh)
    keys = ["full_name", "description", "homepage", "html_url", "default_branch", "created_at",
            "updated_at", "pushed_at", "size", "language", "stargazers_count", "forks_count",
            "watchers_count", "open_issues_count", "subscribers_count", "network_count", "fork",
            "archived", "disabled", "visibility", "has_issues", "has_wiki", "has_pages",
            "has_discussions", "has_downloads", "topics", "is_template"]
    for key in keys:
        if key in repo:
            print(f"{key}: {repo[key]!r}")
    lic = repo.get("license") or {}
    print(f"license: {lic.get('spdx_id')!r} ({lic.get('name')!r})")
    owner = repo.get("owner") or {}
    print(f"owner: {owner.get('login')!r} type={owner.get('type')!r}")
    parent = repo.get("parent") or {}
    if parent:
        print(f"parent: {parent.get('full_name')!r}")
    labels = ["releases", "branches", "tags"]
    for label, path in zip(labels, argv[1:]):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            print(f"{label}: unexpected answer {str(data)[:200]}")
            continue
        print(f"{label}: {len(data)}")
        for item in data:
            if label == "releases":
                assets = [(a.get("name"), a.get("size")) for a in item.get("assets", [])]
                print(f"  {item.get('tag_name')!r} {item.get('name')!r} {item.get('published_at')} assets={assets}")
            elif label == "branches":
                print(f"  {item.get('name')!r} {((item.get('commit') or {}).get('sha'))} protected={item.get('protected')}")
            else:
                print(f"  {item.get('name')!r} {((item.get('commit') or {}).get('sha'))}")
    return 0


def main(argv):
    if len(argv) < 3 or argv[1] not in ("tree", "meta"):
        print(__doc__)
        return 2
    if argv[1] == "tree":
        return cmd_tree(argv[2:])
    return cmd_meta(argv[2:])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
