"""Reading NRRD volumes. Uses `pynrrd` when installed, otherwise a small built-in reader
(supports raw / gzip encoded .nrrd files with attached data)."""
import gzip
import re

import numpy as np

_TYPES = {
    "short": "<i2", "int16": "<i2", "signed short": "<i2", "int16_t": "<i2",
    "ushort": "<u2", "uint16": "<u2", "unsigned short": "<u2", "uint16_t": "<u2",
    "uchar": "<u1", "uint8": "<u1", "unsigned char": "<u1", "uint8_t": "<u1",
    "signed char": "<i1", "int8": "<i1", "int8_t": "<i1",
    "int": "<i4", "int32": "<i4", "signed int": "<i4", "int32_t": "<i4",
    "uint": "<u4", "uint32": "<u4", "unsigned int": "<u4", "uint32_t": "<u4",
    "float": "<f4", "double": "<f8",
}


def _read_minimal(path):
    with open(path, "rb") as f:
        raw = f.read()
    cuts = [i for i in (raw.find(b"\n\n"), raw.find(b"\r\n\r\n")) if i >= 0]
    if not cuts:
        raise ValueError(f"{path}: not a valid NRRD file (no header end)")
    cut = min(cuts)
    sep = 4 if raw[cut:cut + 4] == b"\r\n\r\n" else 2
    header_txt = raw[:cut].decode("ascii", "ignore")
    body = raw[cut + sep:]

    hdr = {}
    for ln in header_txt.splitlines()[1:]:
        if not ln.strip() or ln.startswith("#"):
            continue
        sepi = ln.find(":=") if ":=" in ln else ln.find(":")
        if sepi < 0:
            continue
        k, v = ln[:sepi].strip().lower(), ln[sepi:].lstrip(":=").strip()
        hdr[k] = v

    if "data file" in hdr or "datafile" in hdr:
        raise ValueError("Detached NRRD data (.nhdr) needs pynrrd:  pip install pynrrd")
    enc = hdr.get("encoding", "raw").lower()
    if enc in ("gzip", "gz"):
        body = gzip.decompress(body)
    elif enc != "raw":
        raise ValueError(f"NRRD encoding '{enc}' needs pynrrd:  pip install pynrrd")

    dt = np.dtype(_TYPES[hdr["type"].lower()])
    if hdr.get("endian", "little").lower() == "big":
        dt = dt.newbyteorder(">")
    sizes = [int(x) for x in hdr["sizes"].split()]
    n = int(np.prod(sizes))
    arr = np.frombuffer(body, dtype=dt, count=n).reshape(sizes[::-1]).T   # -> shape == sizes (x, y, z)
    return arr, hdr


def read_nrrd(path):
    """Returns (array with axes (x, y, z), header dict)."""
    try:
        import nrrd  # pynrrd
        data, hdr = nrrd.read(path)
        return data, hdr
    except ImportError:
        return _read_minimal(path)


def get_spacing(hdr):
    """Voxel spacing in mm as a list (x, y, z) or None."""
    try:
        sd = hdr.get("space directions")
        if sd is not None:
            if isinstance(sd, str):
                vecs = [[float(x) for x in v.split(",")] for v in re.findall(r"\(([^)]*)\)", sd)]
            else:
                vecs = [list(v) for v in np.asarray(sd, dtype=float)]
            sp = [float(np.linalg.norm(v)) for v in vecs if not np.isnan(v).any()]
            if sp:
                return sp
        sp = hdr.get("spacings")
        if sp is not None:
            vals = sp.split() if isinstance(sp, str) else list(sp)
            return [float(x) for x in vals]
    except Exception:
        pass
    return None
