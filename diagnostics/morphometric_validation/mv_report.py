"""Render the morphometric validation framework results as a self-contained HTML dashboard.

No external assets, no CDN -- opens offline from the cluster or after scp. Inline SVG charts.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

TRAIT_NOTE = {
    "HW": "Head width. Type III max mediolateral extent of the head part, after rigid alignment.",
    "HL": "Head length. Anterior clypeal margin to occipital margin midpoint.",
    "ML": "Mandible length. Apex to clypeus (convention O2 unresolved).",
    "SL": "Scape length. Antennal insertion to scape apex.",
    "WL": "Weber's length. Both endpoints Type III constructed extremes.",
    "PetL": "Petiole length, ventral outline.",
    "GL": "Gaster length, includes postpetiole where present.",
    "FL": "Hind femur, measured JOINT-TO-JOINT (rotation pivots, not surface).",
    "TBL": "Total body length (derived sum: ML+HL+WL+PetL+GL).",
}


def bar_svg(rows, width=560, rowh=26, vmax=None, unit="%", good=None, warn=None):
    """rows: [(label, value, colour)] -> horizontal bar chart SVG."""
    vmax = vmax or max(v for _, v, _ in rows) * 1.15
    h = rowh * len(rows) + 26
    parts = [f'<svg viewBox="0 0 {width} {h}" width="100%" role="img">']
    lab_w, pad = 62, 8
    plot_w = width - lab_w - 76
    if good is not None:
        gx = lab_w + pad + plot_w * (good / vmax)
        parts.append(f'<line x1="{gx:.1f}" y1="4" x2="{gx:.1f}" y2="{h-22}" '
                     f'stroke="#2e7d32" stroke-width="1" stroke-dasharray="3 3"/>')
    if warn is not None:
        wx = lab_w + pad + plot_w * (warn / vmax)
        parts.append(f'<line x1="{wx:.1f}" y1="4" x2="{wx:.1f}" y2="{h-22}" '
                     f'stroke="#c62828" stroke-width="1" stroke-dasharray="3 3"/>')
    for i, (lab, val, col) in enumerate(rows):
        y = i * rowh + 6
        bw = max(1.0, plot_w * (val / vmax))
        parts.append(f'<text x="{lab_w}" y="{y+13}" text-anchor="end" font-size="12" '
                     f'font-family="ui-monospace,monospace" fill="#333">{lab}</text>')
        parts.append(f'<rect x="{lab_w+pad}" y="{y+2}" width="{bw:.1f}" height="{rowh-10}" '
                     f'rx="2" fill="{col}"/>')
        parts.append(f'<text x="{lab_w+pad+bw+6}" y="{y+13}" font-size="11" '
                     f'font-family="ui-monospace,monospace" fill="#555">{val:.3f}{unit}</text>')
    if good is not None:
        parts.append(f'<text x="{lab_w+pad+plot_w*(good/vmax):.1f}" y="{h-8}" font-size="9" '
                     f'fill="#2e7d32" text-anchor="middle">{good}{unit}</text>')
    if warn is not None:
        parts.append(f'<text x="{lab_w+pad+plot_w*(warn/vmax):.1f}" y="{h-8}" font-size="9" '
                     f'fill="#c62828" text-anchor="middle">{warn}{unit}</text>')
    parts.append("</svg>")
    return "".join(parts)


def ci_svg(rows, width=620, rowh=26, lo=0.6, hi=1.45, null=1.0, ref=None):
    """rows: [(label, exp, ci_lo, ci_hi, colour)] -> exponent + CI whisker chart."""
    h = rowh * len(rows) + 30
    lab_w, pad = 62, 8
    plot_w = width - lab_w - 90

    def X(v):
        return lab_w + pad + plot_w * ((v - lo) / (hi - lo))

    p = [f'<svg viewBox="0 0 {width} {h}" width="100%" role="img">']
    p.append(f'<line x1="{X(null):.1f}" y1="2" x2="{X(null):.1f}" y2="{h-24}" stroke="#555" '
             f'stroke-width="1.5"/>')
    p.append(f'<text x="{X(null):.1f}" y="{h-10}" font-size="9" text-anchor="middle" '
             f'fill="#555">isometry {null}</text>')
    if ref:
        p.append(f'<line x1="{X(ref):.1f}" y1="2" x2="{X(ref):.1f}" y2="{h-24}" stroke="#1565c0" '
                 f'stroke-width="1.5" stroke-dasharray="4 3"/>')
        p.append(f'<text x="{X(ref):.1f}" y="{h-10}" font-size="9" text-anchor="middle" '
                 f'fill="#1565c0">published {ref}</text>')
    for i, (lab, e, cl, ch, col) in enumerate(rows):
        y = i * rowh + 6 + (rowh - 10) / 2
        p.append(f'<text x="{lab_w}" y="{y+4}" text-anchor="end" font-size="12" '
                 f'font-family="ui-monospace,monospace" fill="#333">{lab}</text>')
        p.append(f'<line x1="{X(max(cl,lo)):.1f}" y1="{y}" x2="{X(min(ch,hi)):.1f}" y2="{y}" '
                 f'stroke="{col}" stroke-width="2"/>')
        for xv in (max(cl, lo), min(ch, hi)):
            p.append(f'<line x1="{X(xv):.1f}" y1="{y-4}" x2="{X(xv):.1f}" y2="{y+4}" '
                     f'stroke="{col}" stroke-width="2"/>')
        p.append(f'<circle cx="{X(e):.1f}" cy="{y}" r="4" fill="{col}"/>')
        p.append(f'<text x="{width-84}" y="{y+4}" font-size="11" '
                 f'font-family="ui-monospace,monospace" fill="#555">{e:.3f}</text>')
    p.append("</svg>")
    return "".join(p)


def ba_svg(row, width=560, height=210):
    """Bland-Altman: difference vs mean, with bias and 95% limits of agreement."""
    mean = row["bland_altman"]["mean"]; diff = row["bland_altman"]["diff"]
    lo, hi = min(mean), max(mean)
    span = (hi - lo) or 1.0
    lo -= span * 0.08; hi += span * 0.08
    dvals = diff + [row["loa_lower"], row["loa_upper"], 0.0]
    dlo, dhi = min(dvals), max(dvals)
    dspan = (dhi - dlo) or 1.0
    dlo -= dspan * 0.15; dhi += dspan * 0.15
    L, Rm, T, B = 58, 14, 12, 30
    pw, ph = width - L - Rm, height - T - B
    X = lambda v: L + pw * (v - lo) / (hi - lo)          # noqa: E731
    Y = lambda v: T + ph * (1 - (v - dlo) / (dhi - dlo))  # noqa: E731
    p = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    p.append(f'<rect x="{L}" y="{T}" width="{pw}" height="{ph}" fill="#fbfcfd" stroke="#e3e6ea"/>')
    for val, col, dash, lab in ((0.0, "#999", "2 3", "0"),
                                (row["bias"], "#1565c0", "", "bias"),
                                (row["loa_lower"], "#c62828", "5 4", "−1.96 SD"),
                                (row["loa_upper"], "#c62828", "5 4", "+1.96 SD")):
        y = Y(val)
        p.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}" stroke="{col}" '
                 f'stroke-width="1.2" stroke-dasharray="{dash}"/>')
        p.append(f'<text x="{L-5}" y="{y+3.5:.1f}" text-anchor="end" font-size="9" '
                 f'fill="{col}" font-family="ui-monospace,monospace">{val:+.4f}</text>')
    for mx, dy in zip(mean, diff):
        p.append(f'<circle cx="{X(mx):.1f}" cy="{Y(dy):.1f}" r="3.2" fill="#1565c0" '
                 f'fill-opacity="0.55"/>')
    p.append(f'<text x="{L+pw/2:.0f}" y="{height-8}" text-anchor="middle" font-size="10" '
             f'fill="#666">mean of model and reference →</text>')
    p.append(f'<text x="12" y="{T+ph/2:.0f}" font-size="10" fill="#666" '
             f'transform="rotate(-90 12 {T+ph/2:.0f})" text-anchor="middle">model − reference</text>')
    p.append("</svg>")
    return "".join(p)


def colour_for(v, good, warn):
    return "#2e7d32" if v <= good else ("#e8a33d" if v <= warn else "#c62828")


AG_SECTION = """
<h2>B3b — measurement agreement against the physical reference</h2>
<div class="keybox"><b>Why this is separate from the allometry check above.</b> "Reproduces the
expected biological scaling" is evidence of validity but is <i>not</i> a measurement-accuracy test:
R²=0.99 coexists happily with large systematic bias. Morphometric methodology treats biological
signal and measurement error as separate things, established separately. This section measures
<b>agreement</b> — bias, limits of agreement, proportional bias, Lin's concordance — not correlation.</div>
<div class="warnbox"><b>Read the circularity column first.</b> Model traits live in a per-specimen
normalised frame, so physical scale is restored with <code>scale = BL<sub>mm</sub>/BL<sub>model</sub></code>.
That factor comes from body length, so <b>BL agreement is exactly circular</b> (an arithmetic check,
not evidence) and <b>HW-in-mm is partially circular</b>. The <b>dimensionless HW/BL ratio</b> is the
only fully independent test — the factor cancels. That is the row that means something.</div>
<div class="card"><table>
<tr><th>measurement</th><th>bias</th><th>95% LoA</th><th>mean abs rel</th><th>CCC</th>
    <th>proportional bias</th><th>circularity</th></tr>
{AGROWS}
</table></div>
<div class="axes">{ABPLOTS}</div>
<div class="keybox"><b>Three things this surfaced that R² could not.</b>
<b>(1)</b> The two "head width" definitions are <i>not</i> interchangeable — the reporter-bone
distance is essentially unbiased (−0.13%) while the Type III extent carries a systematic
<b>+2.22%</b> bias. <b>(2)</b> Concordance collapses from <b>0.996 → 0.862</b> once size is removed,
so the near-perfect agreement in mm is largely carried by the shared scale factor — the circularity
made quantitative. <b>(3)</b> The scale-free HW/BL ratio shows <b>significant proportional bias</b>
(p=0.044): error grows with specimen size. That is the same defect as the recovered exponent coming
in at 1.205 against the published 1.2352 — the model slightly compresses the allometric
relationship. Two independent methods, one real finding.</div>
"""


def main():
    d = json.load(open(os.path.join(OUT, "mv_results.json")))
    agp = os.path.join(OUT, "mv_agreement.json")
    ag = json.load(open(agp)) if os.path.exists(agp) else None
    b2 = d["B2_pose_invariance"]["recalibrated"]
    b4 = d["B4_asymmetry"]["recalibrated"]
    b3 = d["B3_biological_validity"]["recalibrated"]
    b1 = d["B1_anatomical_validity"]["per_trait"]
    link = d["B4_predicts_B2"]["recalibrated"]
    pc = b3["positive_control_reference_HW_vs_BL"]

    base = [t for t in ("HW", "HL", "ML", "SL", "WL", "PetL", "GL", "FL", "TBL") if t in b2]

    b2rows = sorted([(t, b2[t]["mean"], colour_for(b2[t]["mean"], 0.5, 2.0)) for t in base],
                    key=lambda r: r[1])
    b4rows = sorted([(t, b4[t]["mean"], colour_for(b4[t]["mean"], 3.0, 6.0)) for t in b4],
                    key=lambda r: r[1])
    cirows = []
    for t in sorted(base, key=lambda t: -b3["traits"][t]["body_length_mm"]["exponent"]):
        e = b3["traits"][t]["body_length_mm"]
        allo = e["ci_lo"] > 1.0 or e["ci_hi"] < 1.0
        cirows.append((t, e["exponent"], e["ci_lo"], e["ci_hi"],
                       "#1565c0" if allo else "#888"))


    if ag:
        agrows = "".join(
            f"<tr><td class='tn'><b>{r['trait']}</b></td>"
            f"<td class='num'>{r['bias']:+.4f}<div class='sub'>{r['bias_pct']:+.2f}%</div></td>"
            f"<td class='num'>[{r['loa_lower']:+.4f}, {r['loa_upper']:+.4f}]</td>"
            f"<td class='num'>{r['mean_abs_rel_pct']:.2f}%</td>"
            f"<td class='num'>{r['ccc']:.3f}</td>"
            f"<td class='num'>{'YES p=%.3f' % r['proportional_bias_p'] if r['proportional_bias_present'] else 'no'}</td>"
            f"<td class='sub' style='max-width:210px'>{r['circularity']}</td></tr>"
            for r in ag["rows"])
        abplots = "".join(
            f"<div class='ax'><b>{r['trait']}</b>{ba_svg(r)}</div>"
            for r in ag["rows"] if "ratio" in r["trait"] or "reporter" in r["trait"])
        ag_html = AG_SECTION.replace("{AGROWS}", agrows).replace("{ABPLOTS}", abplots)
    else:
        ag_html = ""

    def verdict(t):
        pose, anat = b2[t]["mean"], b1.get(t, {})
        e = b3["traits"][t]["body_length_mm"]
        allo = e["ci_lo"] > 1.0 or e["ci_hi"] < 1.0
        if anat.get("verdict") == "UNADJUDICATED":
            return ("UNVALIDATABLE", "#8a6d3b",
                    "no expert landmark exists for this structure -- B1 cannot be evaluated at all")
        if anat.get("verdict") == "RIG_JOINT_NOT_SURFACE":
            return ("RIG-ONLY", "#8a6d3b",
                    "measured between rotation pivots, not surface anatomy; G2 found this class "
                    "unreliable")
        if pose <= 0.5 and allo:
            return ("TRUSTED", "#2e7d32", "pose-invariant and carries independent size signal")
        if pose <= 2.0 and allo:
            return ("USABLE", "#e8a33d", "moderate pose sensitivity; report with the caveat")
        if not allo:
            return ("NO INDEPENDENT SIGNAL", "#8a6d3b",
                    "scales isometrically -- adds nothing beyond body length")
        return ("POSE-FRAGILE", "#c62828", "pose sensitivity dominates; do not compare across poses")

    rows_html = []
    for t in base:
        v, col, why = verdict(t)
        e = b3["traits"][t]["body_length_mm"]
        a1 = b1.get(t, {})
        moved = a1.get("worst_moved_pct_diag")
        rows_html.append(f"""
      <tr>
        <td class="tn"><b>{t}</b><div class="sub">{TRAIT_NOTE.get(t,'')}</div></td>
        <td class="num">{b2[t]['mean']:.3f}%<div class="sub">max {b2[t]['max']:.2f}%</div></td>
        <td class="num">{b4[t]['mean']:.2f}%<div class="sub">max {b4[t]['max']:.1f}%</div></td>
        <td class="num">{e['exponent']:.3f}<div class="sub">[{e['ci_lo']:.2f}, {e['ci_hi']:.2f}]</div></td>
        <td class="num">{a1.get('verdict','-')}<div class="sub">{('landmark moved ' + format(moved,'.1f') + '% diag') if moved else ''}</div></td>
        <td><span class="pill" style="background:{col}">{v}</span><div class="sub">{why}</div></td>
      </tr>""" if t in b4 else f"""
      <tr>
        <td class="tn"><b>{t}</b><div class="sub">{TRAIT_NOTE.get(t,'')}</div></td>
        <td class="num">{b2[t]['mean']:.3f}%<div class="sub">max {b2[t]['max']:.2f}%</div></td>
        <td class="num na">n/a<div class="sub">not bilateral</div></td>
        <td class="num">{e['exponent']:.3f}<div class="sub">[{e['ci_lo']:.2f}, {e['ci_hi']:.2f}]</div></td>
        <td class="num">{a1.get('verdict','-')}<div class="sub">{('landmark moved ' + format(moved,'.1f') + '% diag') if moved else ''}</div></td>
        <td><span class="pill" style="background:{col}">{v}</span><div class="sub">{why}</div></td>
      </tr>""")

    link_rows = "".join(
        f"<tr><td class='tn'>{k}</td><td class='num'>{v['n']}</td>"
        f"<td class='num'>{v['pearson_r']:+.3f}</td><td class='num'>{v['pearson_p']:.3f}</td>"
        f"<td class='num'>{v['spearman_r']:+.3f}</td><td class='num'>{v['spearman_p']:.3f}</td></tr>"
        for k, v in link.items())

    html = f"""<!doctype html>
<meta charset="utf-8">
<title>SMILify Morphometric Validation</title>
<style>
 body{{font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f7f9;color:#1c1e21}}
 .wrap{{max-width:1080px;margin:0 auto;padding:28px 22px 60px}}
 h1{{font-size:25px;margin:0 0 4px}} h2{{font-size:17px;margin:34px 0 10px;padding-bottom:5px;border-bottom:2px solid #e3e6ea}}
 .lede{{color:#555;margin:0 0 22px}}
 .card{{background:#fff;border:1px solid #e3e6ea;border-radius:9px;padding:16px 18px;margin:14px 0}}
 table{{border-collapse:collapse;width:100%;font-size:13px}}
 th{{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#666;
     border-bottom:2px solid #e3e6ea;padding:7px 8px}}
 td{{border-bottom:1px solid #eef0f3;padding:9px 8px;vertical-align:top}}
 td.num{{font-family:ui-monospace,monospace;white-space:nowrap}}
 td.na{{color:#aaa}} td.tn{{max-width:290px}}
 .sub{{color:#8a8f98;font-size:11px;margin-top:2px;font-family:-apple-system,sans-serif}}
 .pill{{color:#fff;padding:2px 9px;border-radius:11px;font-size:11px;font-weight:600;white-space:nowrap}}
 .ok{{color:#2e7d32;font-weight:600}} .bad{{color:#c62828;font-weight:600}}
 .warnbox{{background:#fff8e6;border:1px solid #f0d9a0;border-left:4px solid #e8a33d;
           border-radius:6px;padding:12px 15px;margin:14px 0;font-size:13px}}
 .keybox{{background:#eef4ff;border:1px solid #c9dbff;border-left:4px solid #1565c0;
          border-radius:6px;padding:12px 15px;margin:14px 0;font-size:13px}}
 code{{background:#f0f2f5;padding:1px 5px;border-radius:3px;font-size:12px}}
 .axes{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:11px;margin:12px 0}}
 .ax{{background:#fff;border:1px solid #e3e6ea;border-radius:8px;padding:12px 14px}}
 .ax b{{display:block;font-size:12px;letter-spacing:.03em;color:#1565c0;margin-bottom:3px}}
 .ax span{{font-size:12px;color:#555}}
</style>
<div class="wrap">
<h1>SMILify Morphometric Validation</h1>
<p class="lede">A number extracted from an articulated fitted model is <b>not automatically a
morphometric</b>. Four validity axes, which fail independently — corpus:
{d['n_specimens']} <i>Atta vollenweideri</i>, ARM&nbsp;A fit.</p>

<div class="axes">
  <div class="ax"><b>B1 ANATOMICAL</b><span>Do the endpoints sit on the structure the trait is
    named after? Needs expert landmarks.</span></div>
  <div class="ax"><b>B2 POSE INVARIANCE</b><span>Does it hold constant when only articulation
    changes, shape fixed? Needs nothing but a fit.</span></div>
  <div class="ax"><b>B3 BIOLOGICAL</b><span>Does it reproduce a known scaling relationship?
    Needs reference measurements.</span></div>
  <div class="ax"><b>B4 CONSISTENCY</b><span>Bilateral asymmetry |R−L|/mean. Needs
    <i>no ground truth</i> — computable on all 757.</span></div>
</div>

<div class="keybox"><b>Why four axes and not one.</b> A measurement can be perfectly pose-stable and
still anatomically wrong — measuring the wrong structure, consistently. That is exactly what R3/R4
found (34.7% → 0.2% anatomical error the moment supervision names the correct part, with surface
error unchanged at 0.2% either way). Any single axis certifies fits that are wrong.</div>

<h2>The trust matrix</h2>
<div class="card"><table>
<tr><th>trait</th><th>B2 pose Δ</th><th>B4 asymmetry</th><th>B3 exponent ~BL</th>
    <th>B1 landmark status</th><th>verdict</th></tr>
{''.join(rows_html)}
</table></div>

<h2>B2 — pose invariance</h2>
<p class="lede">Same specimen, joint rotation zeroed, <b>betas / per-joint scale / per-joint
translation / deform_verts held exactly as fitted</b>. Only articulation varies, so the difference
is pure pose contamination. Green ≤0.5%, red >2%.</p>
<div class="card">{bar_svg(b2rows, good=0.5, warn=2.0)}</div>

<h2>B3 — biological validity</h2>
<div class="warnbox"><b>Read the null as 1.0, not 0.</b> <code>load_meshes</code> normalises every
specimen independently, so model-unit traits carry no absolute size. Physical scale is restored per
specimen via <code>BL<sub>mm</sub>/BL<sub>model</sub></code>; because that factor contains body
length, a trait with no independent size information lands at exponent <b>≈1.0</b>. Deviation from
1.0 is the signal — not R², which is inflated by the shared factor.</div>
<div class="card">
<p style="margin:0 0 8px"><b>Positive control:</b> reference head width vs reference body length
(both directly measured, no model involved) → exponent
<span class="{'ok' if pc['passes'] else 'bad'}">{pc['exponent']:.4f}</span>
[{pc['ci_lo']:.3f}, {pc['ci_hi']:.3f}], R²={pc['r2']:.4f}, published <b>{pc['published']}</b> —
<span class="{'ok' if pc['passes'] else 'bad'}">{'PASSES' if pc['passes'] else 'FAILS'}</span>.
If this control fails, nothing below is trustworthy.</p>
{ci_svg(cirows, null=1.0, ref=1.2352)}
<p class="sub" style="margin-top:6px">Blue = CI excludes isometry (independent size signal);
grey = isometric. Dashed blue = the published <i>Atta</i> head-width exponent.</p>
</div>

{ag_html}

<h2>B4 — internal consistency, and can it stand in for the rest?</h2>
<p class="lede">Ants are near-symmetric, so measured left–right disagreement is pipeline error, not
biology. This is the only axis needing no ground truth, so it is the only candidate for a
deployable confidence criterion on an unannotated corpus.</p>
<div class="card">{bar_svg(b4rows, good=3.0, warn=6.0)}</div>
<div class="card">
<p style="margin:0 0 8px"><b>Does B4 predict B2, per specimen, within a trait?</b></p>
<table><tr><th>trait</th><th>n</th><th>pearson r</th><th>p</th><th>spearman r</th><th>p</th></tr>
{link_rows}</table>
<div class="warnbox" style="margin-bottom:0"><b>No — not per specimen.</b> Every correlation is
non-significant (p ≥ 0.10). Asymmetry does <i>not</i> tell you whether a given specimen's
measurement is pose-contaminated. It does rank correctly <b>across traits</b> (SL is worst on both
axes), so B4 is a usable <i>trait-level screen</i> and <b>not</b> a per-specimen confidence score.
Reporting it as the latter would be the exact proxy-without-mechanism error this project has hit
seven times.</div>
</div>

<h2>B1 — anatomical validity</h2>
<div class="warnbox"><b>The traits are still computed at unadjudicated landmarks.</b>
<code>trait_extract.py</code> reads <code>landmark_template_indices.json</code>, which is
rule-computed and explicitly <code>source:"verify"</code> — never anatomically adjudicated. The
expert-recalibrated file moves those exact vertices by <b>7.8–14.7% of the body diagonal</b>
(SL worst at 14.65%). Every trait number computed today carries that offset.
<code>petiole_*</code> and <code>gaster_apex_mid</code> have <b>no</b> expert annotation at all, so
PetL / GL / TBL cannot be validated on this axis even in principle.</div>

<h2>Where this sits</h2>
<div class="card"><table>
<tr><th>file</th><th>role</th></tr>
<tr><td class="tn"><code>diagnostics/morphometric_validation/mv_framework.py</code></td>
    <td>computes all four axes → <code>out/mv_results.json</code></td></tr>
<tr><td class="tn"><code>diagnostics/morphometric_validation/mv_report.py</code></td>
    <td>renders this page</td></tr>
<tr><td class="tn"><code>diagnostics/groundtruth/trait_extract.py</code></td>
    <td>the GLAD trait protocol being validated (reused, not redefined)</td></tr>
<tr><td class="tn"><code>diagnostics/atta_reference/</code></td>
    <td>the biological anchor: 20 Atta, physical reference measurements</td></tr>
<tr><td class="tn"><code>diagnostics/groundtruth/RESULTS_R3/R4/R9_*.md</code></td>
    <td>why B1 exists: anatomy is invisible to surface error, and the objective prefers the wrong
        anatomy</td></tr>
</table></div>
</div>
"""
    path = os.path.join(OUT, "morphometric_validation_report.html")
    with open(path, "w") as fh:
        fh.write(html)
    print("wrote", path)


if __name__ == "__main__":
    main()
