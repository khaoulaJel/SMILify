# Decision — B1 architecture sequencing: build Variant 1 (DensePose-style) first (2026-08-25)

**Status: recommendation for review, not yet acted on.** Phase B's actual build/training work is
still gated on your go-ahead (per the earlier scope decision to hold at Phase A) — this document
only resequences the plan for when that gate opens, using evidence that didn't exist when the
original B1 spec was written.

## The decision

The original handoff spec treats Variant 1 (DensePose-style: discrete leg/segment classification
head + continuous within-segment regression head) and Variant 2 (CSE-style: single continuous
per-point embedding, nearest-neighbor matched) as equally-first-priority, to be built in parallel.
**Build and validate Variant 1 first. Hold Variant 2 in reserve, built only if Variant 1's
part-boundary seam artifacts turn out to be a measured, real problem — not built in parallel from
day one.**

## Why A3 changes this from "equal priority" to "sequenced"

Variant 1's two heads are a direct, mechanical exploitation of exactly the two things A3 just
measured directly on `sa1`/`sa2`, frozen, on a network that was never trained for this task:

| A3 measurement | Variant 1 head it validates |
|---|---|
| Leg-identity linear-probe accuracy: sa1 0.850, sa2 0.866 (7-way, chance ~0.14) | classification head (which leg/segment) |
| Within-leg chain-position R²: sa1 0.838, sa2 0.706 | regression head (position within segment) |

This is a categorically stronger form of evidence than "the architecture should work in
principle" (PHASE10_DESIGN §3's diagnosis of `sa3`'s collapse, or the DensePose/CSE literature
precedents in `LITERATURE_SWEEP_20260825.md`, finding #12): those established that decoding from
`sa1`/`sa2` instead of `sa3` is the right *class* of fix. A3 establishes that the specific
*signal* Variant 1 depends on is measurably present at that point in the network, today, before
any correspondence-specific training. Variant 2 exists specifically to route around a failure
mode (part-boundary seams) that has not been observed yet — it is a hedge against a risk, not a
response to a demonstrated one. Building it in parallel spends effort pre-emptively hedging a risk
that may not materialize, instead of first finding out whether it does.

## What would overturn this

- Variant 1, once built and trained (not just linearly probed), shows seam artifacts at
  segment boundaries that visibly hurt seg_acc or downstream fit quality — the originally-assumed
  reason for Variant 2 to exist. At that point Variant 2 moves from "reserve" to "active,"
  exactly as originally scoped, just triggered by evidence instead of started pre-emptively.
- A3's linear-probe numbers turn out not to predict a trained head's performance (the caveat
  already stated in `RESULTS_A3_architecture_probe_20260825.md`: this is a lower bound/existence
  argument, not a guarantee). If Variant 1 trains and badly underperforms A3's implied ceiling,
  that itself is informative and should be reported before defaulting to Variant 2 as a fix.

## What stays the same

This does not reduce B1's scope or drop Variant 2 from the plan — both variants stay in
`PHASE10_DESIGN_correspondence_network_20260825.md` §3's remit. It only reorders *when* each is
built, using A3's evidence to prioritize the one with direct measured support first. B2 (training
data: A1's corrected sampler + A2's validated dense labels), B3 (pre-registered success bar), and
Phase C (sampler ablation, multi-candidate reconnection) are unaffected by this resequencing.
