"""Experiment B (diagnostic only, no topology changes): attempt to decompose each tangled
(self-intersecting) boundary loop into candidate simple sub-cycles, using its own self-crossing
segment-index pairs (already computed by export_tangled_loop_geometry.cpp) as candidate "cut
chords". Method (explicitly a heuristic, not a topologically-exact minimal-crossing decomposition):

  1. Treat each detected crossing pair (i, j) (segment i=(i,i+1) nearly crosses segment j=(j,j+1))
     as a candidate chord connecting loop-index i to loop-index j.
  2. Greedily accept chords, smallest index-span first (most local crossings resolved first),
     rejecting any candidate that PARTIALLY overlaps (crosses, in the chord-diagram sense) an
     already-accepted chord. Nested or disjoint chords are both accepted - this is the standard
     "non-crossing chord set" condition used to decompose a cycle-with-diagonals into sub-polygons.
  3. Apply accepted chords smallest-span-first to a circular linked list of the loop's vertex
     indices: each chord (i, j) extracts the arc [i..j] (walking forward along the current link
     structure) as one candidate simple sub-cycle, then splices the list to skip that arc.
     What remains after all accepted chords are applied is the "core" residual cycle.

This does NOT triangulate or modify anything. It only reports whether the tangled curve admits a
plausible decomposition into smaller, less-tangled pieces from its own geometry.
"""
import csv
import numpy as np
from collections import defaultdict

BASE = "/home/nao48500/SMILify/diagnostics/local_repair_experiment_20260817"


def load_tangled(spec):
    verts = defaultdict(dict)  # loop_id -> {idx: (x,y,z)}
    with open(f"{BASE}/{spec}_tangled_verts.csv") as f:
        for row in csv.DictReader(f):
            verts[row['loop_id']][int(row['idx_in_loop'])] = (float(row['x']), float(row['y']), float(row['z']))
    pairs = defaultdict(list)
    with open(f"{BASE}/{spec}_tangled_pairs.csv") as f:
        for row in csv.DictReader(f):
            pairs[row['loop_id']].append((int(row['i']), int(row['j'])))
    return verts, pairs


def chords_cross(c1, c2):
    (a, b), (c, d) = c1, c2
    a, b = min(a, b), max(a, b)
    c, d = min(c, d), max(c, d)
    if len({a, b, c, d}) < 4:
        # shared endpoint: not a strict crossing, but splicing one can strand the other's
        # target node off the traversal path and cause an infinite walk in decompose() - so
        # treat as conflicting too (reject), not just strict interval crossings.
        return True
    # partial overlap = crossing; nested or disjoint = not crossing
    return (a < c < b < d) or (c < a < d < b)


def greedy_noncrossing_chords(pairs, n):
    cands = sorted(set((min(i, j), max(i, j)) for i, j in pairs), key=lambda c: c[1] - c[0])
    accepted = []
    for c in cands:
        if c[1] - c[0] < 2:
            continue  # adjacent indices, not a meaningful cut
        if any(chords_cross(c, a) for a in accepted):
            continue
        accepted.append(c)
    return accepted


def decompose(loop_len, accepted_chords):
    nxt = {i: (i + 1) % loop_len for i in range(loop_len)}
    subcycles = []
    for (i, j) in sorted(accepted_chords, key=lambda c: c[1] - c[0]):
        arc = [i]
        cur = i
        steps = 0
        while cur != j:
            cur = nxt[cur]
            arc.append(cur)
            steps += 1
            if steps > loop_len:
                raise RuntimeError(f"decompose(): chord ({i},{j}) failed to reach target within "
                                    f"{loop_len} steps - traversal bug, aborting instead of hanging")
        if len(arc) >= 3:
            subcycles.append(arc)
        nxt[i] = nxt[j]
    core = [0]
    cur = 0
    seen = {0}
    while True:
        cur = nxt[cur]
        if cur in seen:
            break
        core.append(cur)
        seen.add(cur)
    return subcycles, core


def bbox_diag(pts):
    pts = np.array(pts)
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def report_loop(spec, loop_id, vmap, pair_list, verbose=False):
    n = len(vmap)
    pts = [vmap[i] for i in range(n)]
    parent_diag = bbox_diag(pts)
    accepted = greedy_noncrossing_chords(pair_list, n)
    subcycles, core = decompose(n, accepted)
    sizes = sorted([len(s) for s in subcycles] + [len(core)], reverse=True)
    diags = []
    for s in subcycles + [core]:
        d = bbox_diag([vmap[i] for i in s])
        diags.append(d)
    localized = sum(1 for d in diags if d < 0.5 * parent_diag)
    if verbose:
        print(f"  loop_id={loop_id} n_verts={n} n_crossing_pairs={len(pair_list)} "
              f"n_accepted_chords={len(accepted)} n_subcycles={len(subcycles)}+1core "
              f"parent_bbox_diag={parent_diag:.4g}")
        print(f"    subcycle sizes (incl. core): {sizes}")
        print(f"    subcycle bbox_diags: {[f'{d:.4g}' for d in diags]}")
        print(f"    localized (bbox_diag < 50% of parent): {localized}/{len(diags)}")
    return {
        'loop_id': loop_id, 'n_verts': n, 'n_crossing_pairs': len(pair_list),
        'n_accepted_chords': len(accepted), 'n_subcycles_incl_core': len(subcycles) + 1,
        'parent_bbox_diag': parent_diag, 'sizes': sizes, 'diags': diags,
        'n_localized': localized, 'core_size': len(core),
    }


for spec in ["strumigenys_alberti", "mayriella"]:
    verts, pairs = load_tangled(spec)
    print(f"\n===== {spec}: {len(verts)} tangled loops =====")
    results = []
    for loop_id in sorted(verts.keys(), key=int):
        r = report_loop(spec, loop_id, verts[loop_id], pairs[loop_id], verbose=False)
        results.append(r)

    # aggregate
    n_decomposable = sum(1 for r in results if r['n_accepted_chords'] > 0)
    n_fully_irreducible = sum(1 for r in results if r['n_accepted_chords'] == 0)
    avg_reduction = np.mean([r['core_size'] / r['n_verts'] for r in results])
    print(f"loops with >=1 accepted (non-crossing) splitting chord: {n_decomposable}/{len(results)}")
    print(f"loops with NO valid splitting chord found (irreducible by this method): {n_fully_irreducible}/{len(results)}")
    print(f"mean core-residual fraction (core_size/n_verts) after decomposition: {avg_reduction:.3f}")

    # deep dive: Strumigenys patch 1 = loop_id 1
    if spec == "strumigenys_alberti":
        print("\n--- DEEP DIVE: Strumigenys patch 1 (loop_id=1, 1015 verts, 1352 crossing pairs) ---")
        report_loop(spec, "1", verts["1"], pairs["1"], verbose=True)

    # top 5 largest tangled loops for context
    print("\ntop 5 largest tangled loops:")
    for r in sorted(results, key=lambda r: -r['n_verts'])[:5]:
        print(f"  loop_id={r['loop_id']:>4} n_verts={r['n_verts']:>5} n_crossing_pairs={r['n_crossing_pairs']:>5} "
              f"n_accepted_chords={r['n_accepted_chords']:>4} n_subcycles={r['n_subcycles_incl_core']:>3} "
              f"core_size={r['core_size']:>5} core_fraction={r['core_size']/r['n_verts']:.3f}")
