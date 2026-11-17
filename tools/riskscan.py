#!/usr/bin/env python3
"""Static risk-pattern scan of an untrusted checkout (read-only, nothing executed).

Usage: python3 -I riskscan.py <repo-root> [--category NAME] [--max-per-cat N] [--counts]

Prints "<category>\t<relpath>:<line>\t<trimmed line>" for every match. Files that
look binary, and bulky generated/model files, are skipped (listed in SKIP_EXT /
SKIP_NAMES). Patterns are deliberately broad: a hit is a place to read, not a
finding.
"""
import os
import re
import sys

SKIP_DIRS = {".git"}
SKIP_EXT = {".urdf", ".obj", ".stl", ".so", ".npy", ".npz", ".lock"}
SKIP_NAMES = set()

CATEGORIES = [
    ("download", re.compile(
        r"\b(curl|wget)\b|urlretrieve|urllib\.request|requests\.(get|post)|"
        r"hf_hub_download|snapshot_download|from_pretrained|torch\.hub|gdown|"
        r"git\s+clone|pip\s+install|pip3\s+install|apt(-get)?\s+install|dnf\s+install|"
        r"conda\s+(env\s+create|install)|mamba\s+(env\s+create|install)|uv\s+(sync|pip|add)|"
        r"docker\s+(build|run|pull)|FetchContent|ExternalProject|file\(DOWNLOAD|"
        r"download_url|weights\s*=|IMAGENET1K", re.I)),
    ("exec", re.compile(
        r"\bsubprocess\b|os\.system|os\.popen|os\.exec[lv]p?e?\b|\beval\(|\bexec\(|"
        r"shell\s*=\s*True|importlib\.import_module|__import__\(|\bctypes\b|"
        r"execute_process|multiprocessing|Popen")),
    ("deserialize", re.compile(
        r"pickle\.(load|loads|Unpickler)|\bcPickle\b|torch\.load|allow_pickle|"
        r"yaml\.load\(|yaml\.unsafe_load|yaml\.full_load|marshal\.loads?|"
        r"joblib\.load|dill\.|np\.load\(|numpy\.load\(|weights_only")),
    ("credentials", re.compile(
        r"api[_-]?key|apikey|secret|passw(or)?d|\btoken\b|WANDB_API_KEY|HF_TOKEN|"
        r"HUGGING_FACE_HUB_TOKEN|AWS_(ACCESS|SECRET)|BEGIN [A-Z ]*PRIVATE KEY|ssh-rsa|"
        r"ssh-ed25519|ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|\bnetrc\b|twine", re.I)),
    ("telemetry", re.compile(
        r"\bwandb\b|sentry|posthog|telemetry|analytics|mlflow|comet_ml|tensorboard|"
        r"RERUN_ANALYTICS|rerun\s+analytics", re.I)),
    ("network", re.compile(
        r"0\.0\.0\.0|websockets?\.(serve|connect)|\.bind\(|\.listen\(|socket\.socket|"
        r"\bzmq\b|robotmq|tcp://|ws://|wss://|http\.server|\bflask\b|fastapi|uvicorn|"
        r"remote_host|remote_port|--port\b|--host\b", re.I)),
    ("privileged", re.compile(
        r"\bsudo\b|setcap|\bchmod\b|\bchown\b|udevadm|/etc/udev|systemctl|"
        r"/dev/cpu_dma_latency|/dev/shm|\busermod\b|\bgroupadd\b|cap_net|"
        r"sched_setscheduler|SCHED_FIFO|mlockall|os\.nice|setpriority")),
    ("delete", re.compile(
        r"rm\s+-rf?\b|shutil\.rmtree|os\.remove\(|os\.unlink\(|\.unlink\(|os\.rmdir\(|"
        r"rmtree\(")),
    ("machine_specific", re.compile(
        r"/home/[A-Za-z0-9_.-]+|/Users/[A-Za-z0-9_.-]+|/mnt/[A-Za-z0-9_.-]+|/media/[A-Za-z0-9_.-]+|"
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
]


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
    only = None
    max_per = None
    counts_only = "--counts" in argv
    if "--category" in argv:
        only = argv[argv.index("--category") + 1]
    if "--max-per-cat" in argv:
        max_per = int(argv[argv.index("--max-per-cat") + 1])
    cats = [(n, r) for n, r in CATEGORIES if only in (None, n)]
    totals = {n: 0 for n, _ in cats}
    per_file = {n: {} for n, _ in cats}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            ext = os.path.splitext(name)[1].lower()
            if ext in SKIP_EXT or name in SKIP_NAMES:
                continue
            full = os.path.join(dirpath, name)
            if not is_text(full):
                continue
            rel = os.path.relpath(full, root)
            try:
                with open(full, "r", encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, 1):
                        for cat, rx in cats:
                            if rx.search(line):
                                totals[cat] += 1
                                per_file[cat][rel] = per_file[cat].get(rel, 0) + 1
                                if counts_only:
                                    continue
                                if max_per is not None and totals[cat] > max_per:
                                    continue
                                text = line.strip()
                                if len(text) > 200:
                                    text = text[:200] + " ..."
                                print(f"{cat}\t{rel}:{lineno}\t{text}")
            except OSError:
                continue
    print("---- totals ----")
    for cat, _ in cats:
        print(f"{cat}: {totals[cat]} lines in {len(per_file[cat])} files")
        if counts_only:
            for rel, n in sorted(per_file[cat].items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"    {n:5d}  {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
