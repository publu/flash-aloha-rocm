#!/usr/bin/env python3
"""Inspect .npy / .npz files of an untrusted checkout WITHOUT unpickling anything.

Usage: python3 -I npinspect.py <file> [<file> ...]

For every .npy (or member of an .npz zip) prints the header (dtype, shape,
fortran order). For plain numeric arrays it decodes and prints the values with
struct (no numpy import). For object arrays (pickled payloads) it only
DISASSEMBLES the pickle stream with pickletools and lists the opcodes that
name importable objects (GLOBAL / STACK_GLOBAL / INST / OBJ / REDUCE /
BUILD / NEWOBJ); the pickle is never loaded.
"""
import ast
import io
import pickletools
import struct
import sys
import zipfile

MAGIC = b"\x93NUMPY"


def read_header(buf):
    if not buf.startswith(MAGIC):
        raise ValueError("not an .npy stream")
    major, minor = buf[6], buf[7]
    if major == 1:
        (hlen,) = struct.unpack("<H", buf[8:10])
        start = 10
    else:
        (hlen,) = struct.unpack("<I", buf[8:12])
        start = 12
    header_text = buf[start:start + hlen].decode("latin1")
    header = ast.literal_eval(header_text)  # literal dict only; no code evaluated
    return (major, minor), header, start + hlen


def numeric_values(descr, shape, payload):
    if not isinstance(descr, str):
        return None
    order = descr[0]
    kind = descr[1:]
    fmt = {"f8": "d", "f4": "f", "i8": "q", "i4": "i", "i2": "h", "i1": "b",
           "u8": "Q", "u4": "I", "u2": "H", "u1": "B", "b1": "?"}.get(kind)
    if fmt is None:
        return None
    count = 1
    for dim in shape:
        count *= dim
    endian = "<" if order in "<|=" else ">"
    size = struct.calcsize(endian + fmt)
    if len(payload) < count * size:
        return None
    return list(struct.unpack(endian + fmt * count, payload[:count * size]))


def pickle_summary(payload):
    interesting = {"GLOBAL", "STACK_GLOBAL", "INST", "OBJ", "REDUCE", "BUILD",
                   "NEWOBJ", "NEWOBJ_EX", "EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID"}
    ops = {}
    globals_seen = []
    strings = []
    last_strings = []
    try:
        for opcode, arg, _pos in pickletools.genops(io.BytesIO(payload)):
            ops[opcode.name] = ops.get(opcode.name, 0) + 1
            if opcode.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "SHORT_BINSTRING", "BINSTRING", "STRING"):
                last_strings.append(str(arg))
                last_strings = last_strings[-2:]
                if len(strings) < 40:
                    strings.append(str(arg))
            if opcode.name == "GLOBAL":
                globals_seen.append(str(arg))
            if opcode.name == "STACK_GLOBAL":
                globals_seen.append(" ".join(last_strings))
    except Exception as exc:  # malformed stream: report, do not raise
        return {"error": repr(exc), "ops": ops, "globals": globals_seen, "strings": strings}
    return {"ops": {k: v for k, v in ops.items() if k in interesting},
            "all_op_count": sum(ops.values()), "globals": globals_seen, "strings": strings}


def describe(name, buf):
    try:
        version, header, offset = read_header(buf)
    except Exception as exc:
        print(f"  {name}: not parseable as npy ({exc}); {len(buf)} bytes")
        return
    descr = header.get("descr")
    shape = tuple(header.get("shape", ()))
    payload = buf[offset:]
    print(f"  {name}: npy v{version[0]}.{version[1]} descr={descr!r} shape={shape} "
          f"fortran={header.get('fortran_order')} payload={len(payload)} bytes")
    if descr == "|O" or (isinstance(descr, str) and descr.endswith("O")):
        info = pickle_summary(payload)
        print("    OBJECT ARRAY (pickled payload; NOT loaded). Disassembly summary:")
        print(f"    opcodes of interest: {info.get('ops')}  total ops: {info.get('all_op_count')}")
        print(f"    importable names referenced: {info.get('globals')}")
        print(f"    first strings: {info.get('strings')}")
        if "error" in info:
            print(f"    disassembly error: {info['error']}")
        return
    values = numeric_values(descr, shape, payload)
    if values is None:
        if isinstance(descr, str) and ("U" in descr or "S" in descr):
            try:
                width = int(descr[2:])
                if "U" in descr:
                    text = payload.decode("utf-32-le" if descr[0] in "<|=" else "utf-32-be", "replace")
                    items = [text[i:i + width].rstrip("\x00") for i in range(0, len(text), width)]
                else:
                    items = [payload[i:i + width].rstrip(b"\x00").decode("latin1") for i in range(0, len(payload), width)]
                print(f"    strings: {items}")
            except Exception as exc:
                print(f"    (string decode failed: {exc})")
        else:
            print("    (values not decoded)")
        return
    if len(values) <= 64:
        rounded = [round(v, 6) if isinstance(v, float) else v for v in values]
        print(f"    values: {rounded}")
    else:
        print(f"    {len(values)} values; first 8: {values[:8]}")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    for path in argv[1:]:
        print(path)
        try:
            with open(path, "rb") as fh:
                buf = fh.read()
        except OSError as exc:
            print(f"  unreadable: {exc}")
            continue
        if buf.startswith(b"PK\x03\x04"):
            try:
                with zipfile.ZipFile(io.BytesIO(buf)) as zf:
                    for info in zf.infolist():
                        print(f"  zip member {info.filename!r}: {info.file_size} bytes "
                              f"(compressed {info.compress_size}, method {info.compress_type})")
                        if info.file_size > 16 * 1024 * 1024:
                            print("    (member too large; skipped)")
                            continue
                        describe("    " + info.filename, zf.read(info))
            except Exception as exc:
                print(f"  zip error: {exc}")
        else:
            describe(path.rsplit("/", 1)[-1], buf)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
