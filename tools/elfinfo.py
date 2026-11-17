#!/usr/bin/env python3
"""Describe ELF files of an untrusted checkout with a small pure-Python parser.

Usage: python3 -I elfinfo.py <file.so> [<file.so> ...] [--symbols N] [--strings REGEX]

Nothing is loaded, linked or executed: the file is read as bytes. Prints class,
machine, type, SONAME, NEEDED libraries, RPATH/RUNPATH, whether a .symtab /
debug sections exist, the number of dynamic symbols (defined / undefined), the
GLIBC/GLIBCXX version tags it references, sha256, and optionally the first N
defined dynamic symbol names and printable strings matching REGEX.
"""
import hashlib
import re
import struct
import sys

MACHINES = {3: "x86", 40: "ARM", 62: "x86-64", 183: "AArch64", 243: "RISC-V"}
TYPES = {1: "REL", 2: "EXEC", 3: "DYN (shared object / PIE)", 4: "CORE"}
DT_NEEDED, DT_SONAME, DT_RPATH, DT_RUNPATH = 1, 14, 15, 29


def cstr(blob, offset):
    if offset < 0 or offset >= len(blob):
        return ""
    end = blob.find(b"\x00", offset)
    if end < 0:
        end = len(blob)
    return blob[offset:end].decode("utf-8", "replace")


def parse(path, n_symbols, strings_rx):
    with open(path, "rb") as fh:
        data = fh.read()
    print(path)
    print(f"  size {len(data)} bytes  sha256 {hashlib.sha256(data).hexdigest()}")
    if data[:4] != b"\x7fELF":
        print("  not an ELF file")
        return
    is64 = data[4] == 2
    little = data[5] == 1
    end = "<" if little else ">"
    if not is64:
        print("  32-bit ELF (not parsed further)")
        return
    (e_type, e_machine, _ver, _entry, _phoff, e_shoff, _flags, _ehsize, _phentsize, _phnum,
     e_shentsize, e_shnum, e_shstrndx) = struct.unpack(end + "HHIQQQIHHHHHH", data[16:64])
    print(f"  ELF64 {'LE' if little else 'BE'}  machine {MACHINES.get(e_machine, e_machine)}  "
          f"type {TYPES.get(e_type, e_type)}  sections {e_shnum}")
    sections = []
    for index in range(e_shnum):
        off = e_shoff + index * e_shentsize
        if off + 64 > len(data):
            break
        (sh_name, sh_type, _fl, _addr, sh_offset, sh_size, sh_link, _info, _align,
         sh_entsize) = struct.unpack(end + "IIQQQQIIQQ", data[off:off + 64])
        sections.append(dict(name_off=sh_name, type=sh_type, offset=sh_offset, size=sh_size,
                             link=sh_link, entsize=sh_entsize))
    if not sections or e_shstrndx >= len(sections):
        print("  no section headers")
        return
    shstr = sections[e_shstrndx]
    shstr_blob = data[shstr["offset"]:shstr["offset"] + shstr["size"]]
    for sec in sections:
        sec["name"] = cstr(shstr_blob, sec["name_off"])
    names = [sec["name"] for sec in sections]
    print(f"  has .symtab: {'.symtab' in names}   debug sections: "
          f"{[n for n in names if n.startswith('.debug')][:6]}   .comment: {'.comment' in names}")
    for sec in sections:
        if sec["name"] == ".comment":
            blob = data[sec["offset"]:sec["offset"] + sec["size"]]
            parts = [p.decode("utf-8", "replace") for p in blob.split(b"\x00") if p]
            print(f"  .comment: {parts[:4]}")
    needed, soname, rpath = [], None, []
    for sec in sections:
        if sec["type"] != 6:  # SHT_DYNAMIC
            continue
        strtab = sections[sec["link"]] if sec["link"] < len(sections) else None
        if strtab is None:
            continue
        str_blob = data[strtab["offset"]:strtab["offset"] + strtab["size"]]
        for pos in range(sec["offset"], sec["offset"] + sec["size"], 16):
            if pos + 16 > len(data):
                break
            tag, value = struct.unpack(end + "qQ", data[pos:pos + 16])
            if tag == 0:
                break
            if tag == DT_NEEDED:
                needed.append(cstr(str_blob, value))
            elif tag == DT_SONAME:
                soname = cstr(str_blob, value)
            elif tag in (DT_RPATH, DT_RUNPATH):
                rpath.append(cstr(str_blob, value))
    print(f"  SONAME: {soname}")
    print(f"  NEEDED: {needed}")
    print(f"  RPATH/RUNPATH: {rpath}")
    defined, undefined = [], []
    for sec in sections:
        if sec["type"] != 11:  # SHT_DYNSYM
            continue
        strtab = sections[sec["link"]] if sec["link"] < len(sections) else None
        if strtab is None or not sec["entsize"]:
            continue
        str_blob = data[strtab["offset"]:strtab["offset"] + strtab["size"]]
        for pos in range(sec["offset"], sec["offset"] + sec["size"], sec["entsize"]):
            if pos + 24 > len(data):
                break
            st_name, _info, _other, st_shndx, _value, _size = struct.unpack(end + "IBBHQQ", data[pos:pos + 24])
            name = cstr(str_blob, st_name)
            if not name:
                continue
            (undefined if st_shndx == 0 else defined).append(name)
    print(f"  dynamic symbols: {len(defined)} defined, {len(undefined)} undefined")
    versions = sorted(set(re.findall(rb"(?:GLIBCXX|GLIBC|CXXABI|GCC)_[0-9.]+", data)))
    tops = {}
    for tag in versions:
        family, _, number = tag.decode().partition("_")
        key = tuple(int(part) for part in number.split(".") if part.isdigit())
        if family not in tops or key > tops[family][0]:
            tops[family] = (key, tag.decode())
    print(f"  highest version tags referenced: {sorted(v[1] for v in tops.values())}")
    interesting = [n for n in undefined if re.search(
        r"^(socket|connect|bind|listen|accept|send|sendto|recv|recvfrom|getaddrinfo|gethostbyname|"
        r"system|popen|exec[lv]p?e?|fork|dlopen|dlsym|curl_|SSL_|ioctl|open|fopen|unlink|remove|"
        r"setsockopt|if_nametoindex|write|read|ptrace|mprotect|mmap)$", n)]
    print(f"  undefined libc-style calls of interest: {sorted(set(interesting))}")
    if n_symbols:
        print(f"  first {n_symbols} defined dynamic symbols: {defined[:n_symbols]}")
    if strings_rx:
        rx = re.compile(strings_rx.encode())
        hits = []
        for match in re.finditer(rb"[\x20-\x7e]{6,}", data):
            text = match.group(0)
            if rx.search(text):
                hits.append(text.decode("ascii", "replace")[:160])
        print(f"  strings matching /{strings_rx}/: {len(hits)}")
        for text in hits[:60]:
            print(f"    {text}")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    n_symbols = int(argv[argv.index("--symbols") + 1]) if "--symbols" in argv else 0
    strings_rx = argv[argv.index("--strings") + 1] if "--strings" in argv else None
    skip = set()
    for flag in ("--symbols", "--strings"):
        if flag in argv:
            skip.update({argv.index(flag), argv.index(flag) + 1})
    for index, path in enumerate(argv[1:], 1):
        if index in skip:
            continue
        try:
            parse(path, n_symbols, strings_rx)
        except Exception as exc:  # malformed file: report and continue
            print(f"  parse error: {exc!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
