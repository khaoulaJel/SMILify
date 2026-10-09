import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from hw_core import *
from hw_render import *
M = load_model(); F = np.asarray(M["dd"]["f"], np.int64); V0 = M["v_template"]
axes = orient_frame(M)
R = build_head_width_reporter_matrix(V0).numpy()
PL, PR = R[0] @ V0, R[1] @ V0
fig, AX = plt.subplots(1, 3, figsize=(15, 5))
for a, v in zip(AX, ["dorsal", "lateral", "anterior"]):
    draw_mesh(a, V0, F, axes, v)
    for p, c in ((PL, "red"), (PR, "blue")):
        q = project(p, axes, v); a.scatter(q[:, 0], q[:, 1], s=60, c=c, zorder=10)
    a.set_title(v)
fig.savefig(f"{SHOW}/figures/_probe_orientation.png", dpi=80, bbox_inches="tight"); print("ok")
