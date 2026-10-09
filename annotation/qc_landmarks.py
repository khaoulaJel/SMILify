"""qc_landmarks.py -- validate the placed surface landmarks, and measure the traits they imply.

Checks are framed on the landmarks' OWN geometry (lateral axis from the head-width pair, antero-
posterior from occiput to clypeus), so they are independent of scale and of any frame confusion.

MANDIBULAR APEXES ARE EXEMPT from the left/right test: many ants hold their mandibles crossed or
strongly curved, so the tip of the right jaw legitimately sits on the left. Only the articulation
cannot cross, and that lives in the joint file, not here. Confirmed on Aenictus.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BOUNDS = {"HW/HL": (0.40, 2.00), "SL/HL": (0.20, 3.00), "ML/HL": (0.10, 2.00)}


def load(p):
    d = json.load(open(p))
    L = {k: np.array(v["original"], float) for k, v in d["landmarks"].items()}
    return d["specimen_id"], L, d.get("missing", [])


def frame(L):
    lat = L["head_width_r"] - L["head_width_l"]
    lat /= np.linalg.norm(lat)
    ap = L["clypeal_ant_mid"] - L["cephalic_post_mid"]
    ap -= (ap @ lat) * lat
    ap /= np.linalg.norm(ap)
    return lat, ap, (L["head_width_r"] + L["head_width_l"]) / 2


def main():
    rows, bad = [], 0
    print(f"{'specimen':38s}{'n':>3}{'HW/HL':>7}{'ML/HL':>7}{'SL/HL':>7}{'WL':>9}   issues")
    for p in sorted(glob.glob(os.path.join(HERE, "landmarks", "*_traits.json"))):
        sid, L, missing = load(p)
        # A missing landmark is DATA, not a defect: the structure is genuinely absent on that
        # specimen (Cephalotes atratus has no antennae). Reported separately, never as an issue.
        msgs = []
        need = ["head_width_r", "head_width_l", "clypeal_ant_mid", "cephalic_post_mid"]
        if any(k not in L for k in need):
            print(f"{sid[:37]:38s} cannot build a frame -- skipped")
            continue
        lat, ap, mid = frame(L)
        HL = np.linalg.norm(L["clypeal_ant_mid"] - L["cephalic_post_mid"])
        s = {k: float((v - mid) @ lat) / HL for k, v in L.items()}
        f = {k: float((v - mid) @ ap) / HL for k, v in L.items()}

        for k in ("clypeal_ant_mid", "cephalic_post_mid"):
            if abs(s[k]) > 0.25:
                msgs.append(f"{k} off the midline ({100*s[k]:+.0f}% HL)")
        for a, b in (("antennal_insertion_r", "antennal_insertion_l"),
                     ("scape_apex_r", "scape_apex_l")):
            if a in s and b in s and s[a] <= s[b]:
                msgs.append(f"{a[:-2]} sides inverted ({100*s[a]:+.0f} vs {100*s[b]:+.0f}% HL)")
        for k in ("mandibular_apex_r", "mandibular_apex_l"):
            if k in f and f[k] <= f["clypeal_ant_mid"]:
                msgs.append(f"{k} is behind the clypeal margin")
        for side in "rl":
            a, i = f"scape_apex_{side}", f"antennal_insertion_{side}"
            if a in L and i in L and np.linalg.norm(L[a] - L[i]) / HL < 0.15:
                msgs.append(f"scape_{side} apex sits on its own insertion")

        HW = np.linalg.norm(L["head_width_r"] - L["head_width_l"])
        ML = (np.linalg.norm(L["mandibular_apex_r"] - L["clypeal_ant_mid"])
              if "mandibular_apex_r" in L else np.nan)
        SL = (np.linalg.norm(L["scape_apex_r"] - L["antennal_insertion_r"])
              if "scape_apex_r" in L and "antennal_insertion_r" in L else np.nan)
        WL = (np.linalg.norm(L["wl_anterior_r"] - L["wl_posterior_r"])
              if "wl_anterior_r" in L and "wl_posterior_r" in L else np.nan)
        R = {"HW/HL": HW / HL, "ML/HL": ML / HL, "SL/HL": SL / HL}
        for k, (lo, hi) in BOUNDS.items():
            if R[k] == R[k] and not (lo <= R[k] <= hi):
                msgs.append(f"{k}={R[k]:.2f} outside the biological bound [{lo},{hi}]")

        rows.append(dict(specimen=sid, n=len(L), absent=missing, HL=HL, HW=HW, ML=ML, SL=SL, WL=WL,
                         **{k: (float(v) if v == v else None) for k, v in R.items()},
                         issues=msgs))
        bad += len(msgs)
        print(f"{sid[:37]:38s}{len(L):>3}{R['HW/HL']:>7.2f}{R['ML/HL']:>7.2f}{R['SL/HL']:>7.2f}"
              f"{WL:>9.0f}   {'; '.join(msgs) if msgs else 'clean'}"
              f"{('   [absent: ' + ', '.join(m.replace('_',' ') for m in missing) + ']') if missing else ''}")

    json.dump(rows, open(os.path.join(HERE, "landmark_traits.json"), "w"), indent=1)
    print(f"\n{len(rows)} specimens, {bad} issue(s). "
          f"-> annotation/landmark_traits.json")
    if rows:
        for k in BOUNDS:
            v = np.array([r[k] for r in rows if r[k] is not None])
            print(f"  {k:>6}  median {np.median(v):.2f}   range {v.min():.2f}-{v.max():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
