# F7 — the embedding geometry is fine. The residual hypothesis is closed, and top-1 was the wrong yardstick.

Free diagnostic on the frozen C11 key table (10,235 unit-norm vertex embeddings in 16 dims). No
training, no fitting. Tests the one hypothesis F6's write-up left standing: that 16 dimensions are
too few to separate 10,235 vertices.

| measurement | value | reading |
|---|---|---|
| participation-ratio effective dimension | **14.82 / 16** | the space is nearly fully used, not collapsed |
| top-3 components' variance share | 0.279 | no dominant low-rank structure |
| median cosine to nearest *other* key | 0.9837 | keys **are** crowded |
| fraction with a >0.99-similar neighbour | 0.190 | heavily crowded for a fifth of vertices |
| **graph distance to that nearest key** (/ body diagonal) | **median 0.0049** | … |
| fraction of those further than 0.10 | **0.000** | **the crowding is entirely local** |
| leg vertices whose nearest key is on a *different leg* | **0.0001** | anatomically correct |

## Verdict: capacity is not the bottleneck, and the structure is right

The keys are crowded, but every crowded pair is a pair of vertices that are **adjacent on the
mesh**. Not one vertex in 10,235 has its nearest key more than 0.10 body-diagonals away, and
essentially none has it on a different leg. The embedding has learned exactly the geometry it
should: near in embedding space ⇔ near on the surface.

So the 16-dimensional space is not the constraint, and the residual hypothesis recorded at the end
of F6 is closed — negatively for the hypothesis, positively for the model.

## What this does to the premise the whole F-series was built on

The series was motivated by a gap: information linearly decodable at **52.9°** versus retrieval
**top-1 = 0.071**. F7 says that gap is substantially **not a defect**.

With 19% of vertices having a >0.99-similar neighbour — because 10,235 vertices sit on a small
closed surface, so adjacent ones are genuinely near-identical — exact-vertex top-1 *cannot* be
high, and does not need to be. The retrieved vertex sits a median **0.0049 body-diagonals** from
the true one.

This is the same thing three earlier results said from three directions, now measured directly in
the embedding space itself:

- **C3's own docstring**: exact-vertex accuracy is "harsh on a dense mesh… the 3D error is the one
  to weigh".
- **E0**: the correspondence metric's own floor is 0.0908 on the coxa, because adjacent surface
  points are genuinely ambiguous.
- **F6**: softening the target changed nothing measurable — because the target encoding was never
  what made top-1 low.

**The correct conclusion is that top-1 was the wrong yardstick, not that the model is weak.** F6
tried to fix in the *loss* something that only ever needed fixing in the *evaluation framing*.

## Series close

| id | idea | verdict |
|---|---|---|
| F1 | is the signal in the features? | **yes** — 52.9° vs 90° chance; equivariant retrain not justified |
| F2 | trained 1-of-5 selector | FAIL — rank accuracy is not error reduction |
| F4 | optimal-transport assignment | FAIL (4.3% / 4.5% vs a 5% bar) |
| F5 | conformal fit-quality intervals | FAIL — the GT-free signals do not predict error |
| F6 | geodesic-softened targets | PARTIAL, within run-to-run noise |
| F7 | is 16-d enough? | **not the bottleneck** — crowding is local and anatomically correct |

Five candidate locations for the gap have now been tested and excluded: **backbone architecture,
target encoding, selection, post-hoc assignment, embedding capacity.** The sixth possibility — that
a large part of the apparent gap was never there, being an artefact of scoring exact-vertex
identity on a dense mesh — is the one F7 supports.

**No further algorithmic variant is opened from this.** What limits the correspondence line now is
not a component to improve; it is the absence of ground truth on real specimens, which no
experiment on this side can supply.
