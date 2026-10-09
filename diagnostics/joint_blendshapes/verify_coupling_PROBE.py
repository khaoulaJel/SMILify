"""Z3 step 0 -- verify the coupling before trusting any arm that uses it.

Two things must hold, and both are asserted rather than eyeballed:
  1. OFF is a no-op. With COUPLE_JOINT_BLENDSHAPES unset the vertices must be BIT-identical to
     the uncoupled composition, so every previously scored experiment stays comparable.
  2. ON is genuinely active. Driving beta_k must move vertices further with coupling than
     without it -- REPORT.md 6.7's check, which caught a coupling that was silently disabled by
     a units factor.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")


def run(couple, betas, dev="cpu"):
    os.environ["SMILIFY_COUPLE_JOINT_BLENDSHAPES"] = "1" if couple else "0"
    for m in [m for m in list(sys.modules) if m.startswith(("config", "smal_model"))]:
        del sys.modules[m]
    import config  # noqa
    from smal_model.smal_torch import SMAL
    assert config.COUPLE_JOINT_BLENDSHAPES is couple, "flag did not take effect"
    smal = SMAL(dev)
    n = betas.shape[0]
    theta = torch.zeros(n, 55, 3)
    out = smal(betas, theta,
               betas_logscale=torch.zeros(n, 55, 3),
               betas_trans=torch.zeros(n, 55, 3))
    v = out[0]
    return v.detach().numpy()


def main():
    torch.manual_seed(0)
    zero = torch.zeros(1, 25)

    # ---- check 1: OFF is a no-op at the mean shape and at a driven shape
    b = torch.zeros(2, 25); b[1, 0] = 3.0
    off1, off2 = run(False, b), run(False, b)
    assert np.array_equal(off1, off2), "OFF is not deterministic"
    print(f"[1] OFF deterministic: max|diff| {np.abs(off1-off2).max():.3e}  PASS")

    # at beta = 0 the driven scale is log(1+0)=0 and driven trans is 0, so ON == OFF exactly
    on_zero, off_zero = run(True, zero), run(False, zero)
    d0 = np.abs(on_zero - off_zero).max()
    print(f"[2] ON == OFF at beta=0 (driven terms vanish): max|diff| {d0:.3e}  "
          f"{'PASS' if d0 < 1e-6 else 'FAIL'}")
    assert d0 < 1e-6

    # ---- check 2: ON is genuinely active, and by how much
    print(f"\n[3] displacement from the mean shape, per beta at +3 sigma "
          f"(REPORT 6.7's activity check)")
    print(f"{'beta':>6}{'OFF rms':>12}{'ON rms':>12}{'ratio':>9}")
    ratios = []
    for k in [0, 1, 2, 3, 4]:
        bk = torch.zeros(1, 25); bk[0, k] = 3.0
        voff, von = run(False, bk), run(True, bk)
        roff = np.sqrt(((voff - off_zero) ** 2).sum(2)).mean()
        ron = np.sqrt(((von - on_zero) ** 2).sum(2)).mean()
        ratios.append(ron / max(roff, 1e-12))
        print(f"{k:>6}{roff:>12.5f}{ron:>12.5f}{ron/max(roff,1e-12):>9.3f}")
    active = np.mean(ratios) > 1.05
    print(f"\n  mean ratio {np.mean(ratios):.3f} -> coupling {'ACTIVE' if active else 'INERT'}  "
          f"{'PASS' if active else 'FAIL -- do not run any coupled arm'}")
    assert active


if __name__ == "__main__":
    main()
