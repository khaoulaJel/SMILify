# F1 result — the circumferential signal DOES reach the head. An equivariant retrain is not justified.

Bar fixed in `PREREGISTRATION_F1_circumferential_probe_20260828.md` before the run. Frozen C11
backbone (`best_model.pt`, epoch 119, loaded **strictly**, 207 tensors), ridge probes per
(leg, segment) group, 120 train / 60 held-out specimens from the same split C3/C11 used.
Chance = 90°.

| depth | dim | **circumferential** | axial R² (control) | shuffled (control) |
|---|---|---|---|---|
| `xyz` (raw input) | 3 | 82.18° | +0.398 | 88.50° |
| `sa1` (after 1st max-pool) | 320 | **58.26°** | +0.434 | 89.13° |
| `sa2` (after 2nd max-pool) | 640 | 81.73° | +0.462 | 88.61° |
| `fp1` (what the head consumes) | 128 | **52.89°** | +0.852 | 89.09° |

**Both controls pass.** Axial R² rises monotonically with depth (0.398 → 0.852) — the probe is
sensitive, so a null would have been interpretable. Shuffled targets sit at 88.5–89.1°, i.e. the
90° chance level, at every depth.

## Verdict: SIGNAL IS PRESENT THROUGHOUT

The pre-registered reading for `E(fp1) ≤ 70°` fires: **52.89°**, a 37° margin below chance, on the
exact tensor the embedding head consumes.

> "Then the information reaches the head and the circumferential failure is in the *loss/retrieval*,
> not the architecture — and neither an equivariant retrain nor an attention decoder is justified."

This closes the equivariance question that the LRF check and the invalid rotation-ensemble test left
open. The geometry carries the signal (LRF: `circum_gap` 0.38–0.44), **and so do the features**.
Max-pooling does not destroy it. The failure is downstream of both.

`fp1` beats raw `xyz` (52.9° vs 82.2°) because θ is a nonlinear function of position even inside a
single segment frame; the learned features linearise it. That is real signal, not an artefact of
dimensionality — the shuffle control at the same 128 dims lands at 89.1°.

## The sharpened statement of the problem

C11's own held-out retrieval is **top1 = 0.071**. So circumferential angle is linearly decodable
from `fp1` at 52.9° while the retrieval built on that same tensor is at 7% top-1. **The information
is there and the contrastive head is not using it.** That is a loss/objective problem, not a
representational one — and it is a much cheaper thing to attack than an equivariant backbone.

Note `sa2` sits near chance (81.7°) while `sa1` is at 58.3° and `fp1` at 52.9°: the 128-centroid
bottleneck does lose the direction, and the `fp3`/`fp2`/`fp1` skip path restores it from `sa1`.
The architecture is already routing around its own bottleneck.

## Two defects found and fixed before this run, recorded because neither was visible in the output

1. **The checkpoint silently did not load.** Weights live under `model_state_dict`; the script
   looked for `model`, fell through to the whole dict, and `load_state_dict(strict=False)` matched
   nothing — 178 missing keys, a randomly initialised network. A non-zero-weight assert does not
   catch this, because random weights are non-zero. That run produced a plausible-looking
   "signal at sa1, lost by fp1" pattern that was pure noise. Now `strict=True` plus a
   post-load tensor comparison against the raw file.
2. **Pooling all groups into one probe made the target non-linear** (θ is defined in each segment's
   own PCA frame), which is why the raw-`xyz` control sat at chance. Now one probe per group.

## What is NOT claimed

- A linear probe measures **linear decodability**. 52.89° bounds what a *linear* head extracts; it
  does not prove the contrastive head could use it, only that the information is present in a
  readily accessible form.
- Nothing about real scans — this is the synthetic corpus the backbone was trained on.
- No claim that fixing circumferential retrieval would move `leg_acc`. C11/C12 twice showed
  retrieval gains not reaching the fitter, and F1 does not test the fitter at all.
