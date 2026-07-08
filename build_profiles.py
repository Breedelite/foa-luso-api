#!/usr/bin/env python3
"""Extract per-layer APSoil water profiles from the APSIM Apsim_Soils.xml source
into apsoil_profiles.json, keyed by ApsoilNumber. This is the data behind the
volumetric-water-over-depth chart the dashboard draws live for the selected soil.

Run:  python3 build_profiles.py "/path/to/Apsim_Soils.xml"
Default source: ~/Desktop/CSIRO Ag Productivity/Apsim_Soils.xml
Output: apsoil_profiles.json (next to this script)
"""
import os, sys, json, math
import xml.etree.ElementTree as ET

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Desktop/CSIRO Ag Productivity/Apsim_Soils.xml")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apsoil_profiles.json")


def arr(el, tag, nd=3):
    """Layer values, rounded. Non-finite (NaN/missing) become None so the JSON
    stays valid and the chart can skip those points."""
    e = el.find(tag)
    if e is None:
        return []
    out = []
    for d in e.findall("double"):
        try:
            v = float(d.text)
        except (TypeError, ValueError):
            out.append(None); continue
        out.append(round(v, nd) if math.isfinite(v) else None)
    return out


def build(src):
    profiles = {}
    ctx = ET.iterparse(src, events=("end",))
    for ev, el in ctx:
        if el.tag != "Soil":
            continue
        num = (el.findtext("ApsoilNumber") or "").strip()
        w = el.find("Water")
        if not num or w is None:
            el.clear(); continue
        th = arr(w, "Thickness", 0)
        if not th:
            el.clear(); continue
        dul = arr(w, "DUL")
        crops = {}
        for sc in w.findall("SoilCrop"):
            name = (sc.get("name") or "").strip().lower()
            ll = arr(sc, "LL")
            if not ll:
                continue
            pawc = sum((d - l) * t for d, l, t in zip(dul, ll, th)
                       if d is not None and l is not None)
            crops[name] = {"ll": ll, "pawc": round(pawc)}
        profiles[num] = {
            "soiltype": (el.findtext("SoilType") or "").strip(),
            "datasource": (el.findtext("DataSource") or "").strip(),
            "thickness": [int(x) for x in th],
            "airdry": arr(w, "AirDry"),
            "ll15": arr(w, "LL15"),
            "dul": dul,
            "sat": arr(w, "SAT"),
            "crops": crops,
        }
        el.clear()
    return profiles


if __name__ == "__main__":
    if not os.path.exists(SRC):
        sys.exit(f"Source XML not found: {SRC}")
    profiles = build(SRC)
    with open(OUT, "w") as f:
        json.dump(profiles, f, separators=(",", ":"))
    size = os.path.getsize(OUT)
    print(f"Wrote {len(profiles)} soil profiles -> {OUT} ({size/1024:.0f} KB)")
    print("Sample 424:", json.dumps(profiles.get("424", {}), indent=1)[:400])
