# Pre-registration — F3: mean-teacher self-training on bench50

Written before any F3 code exists, 2026-08-28. This is the last open path capable of producing a
*positive* result on real specimens, which is exactly why it gets more rigor, not less.

## What is actually being tested

Everything validated on real ants so far is **inference only**: the CSE head was trained purely on
synthetic data and run on bench50. It has never been fine-tuned on real scans using the fitter's own
converged output as weak supervision. That is a different lever from hard-negative mining (C11),
which was supervision *pressure* on the same synthetic distribution and failed its own bar.

Method: mean-teacher. Teacher = EMA of the student; pseudo-labels are the fitter's converged
correspondence on real scans; the student trains on augmented views of the same scans. Standard
consistency-regularisation setup, nothing exotic.

## The constraint that shapes the whole design

**bench50 has no correspondence ground truth. `leg_acc`/`seg_acc` cannot be computed on it.**
This is the same limitation that made the original real-scan claims geometric, and it is
non-negotiable — the annotation effort that could change it does not exist yet.

So the endpoint cannot be the metric this investigation actually cares about. Three measurements
are used instead, with sharply different standing, and that difference is fixed here:

1. **Per-segment geometric fit on held-out real specimens** (chamfer/fscore restricted to each
   segment's territory). *Weak.* An improvement here alone is **instance #2 of this
   investigation's decoupling pattern** (bench50's original framing: geometric fit improved with
   correspondence never checked) and is **pre-declared not to count as a correspondence win.**
2. **GT-free correctness proxies on real scans** — bilateral symmetry consistency and
   cycle-consistency. *Medium.* These test correspondence *correctness*, not plausibility.
3. **Synthetic non-degradation** — held-out synthetic retrieval, where GT exists. *Strong, and the
   mechanism check.* See below.

## Mechanism check, decided now rather than invented after a good number

Fine-tuning on unlabelled real data is precisely the setup where a model can satisfy its training
objective by a shortcut — matching plausible-looking regions rather than correct ones — and with no
real GT, nothing in the real-scan metrics would catch it. Two checks, both mandatory, both reported
whatever the endpoint says:

- **Synthetic retrieval must not degrade.** Re-evaluate held-out *synthetic* top-1/top-5 and median
  3D error after fine-tuning. Correspondence knowledge is a property of the network, not of the
  domain; a fine-tune that improves real geometry while destroying synthetic correspondence has
  learned something that is not correspondence. Threshold: synthetic top-1 must not fall by more
  than **10% relative** to C11's 0.0709. A larger drop makes the arm a **SHORTCUT** result
  regardless of anything measured on real scans.
- **Bilateral symmetry consistency.** A real ant is bilaterally symmetric, so a point on the left
  side and its mirrored counterpart should retrieve mirror-paired template vertices. This is
  GT-free and specific to correctness. **Prerequisite probe, to be done first:** `dd["sym_verts"]`
  is only 227 indices (midline vertices), **not** a left↔right pairing, so the mirror map must be
  *built* (reflect `v_template` across the sagittal plane, nearest-neighbour match) and
  *validated* (`mirror(mirror(v)) == v`; midline vertices must map to themselves) before it is
  used. If the map cannot be validated, this check is dropped and the pre-registration says so
  rather than substituting a weaker one.

## Splits and bar

**Real specimens are split at the specimen level** into fine-tune and held-out sets. No specimen
appears in both. Reported on held-out real specimens only.

**Every reading is per segment** (`co`/`tr`/`fe`/`ti`/`ta`/`pt`), never pooled — F2 re-confirmed
that a pooled number hides segment-specific failure, and `ti`/`ta` are currently coverage-only, so
a `tr`/`fe`-driven pooled improvement masking no change on `ti`/`ta`/`co` is the specific failure
mode to prevent.

- **PASS** — the GT-free correctness proxies improve on held-out real specimens, on **at least
  three segments individually**, **and** synthetic retrieval does not degrade past the 10% bound.
  Geometric fit is reported but is not sufficient for a pass on its own.
- **GEOMETRIC-ONLY** — geometric fit improves, correctness proxies do not. Reported as the sixth
  instance of the decoupling pattern, **not** as a win.
- **SHORTCUT** — synthetic retrieval degrades past the bound. Reported as such regardless of how
  good the real-scan numbers look.
- **FAIL** — nothing moves.

## What will NOT be claimed

- Not that fine-tuning "proves correspondence works on real ants". Without real GT, nothing here
  can prove that; only the annotation effort could.
- No `leg_acc`/`seg_acc` number on bench50 will be quoted, because none can be computed.
- No claim about `ti`/`ta` acquiring real correspondence unless the per-segment correctness proxies
  say so for those specific segments.
