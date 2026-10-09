# M4-C — hierarchical variance decomposition: the genus signal is real, and concentrated at genus **for HW only**

Run 2026-09-10 on the frozen M4-A admissible set, per `PREREGISTRATION_M4_corpus_analysis.md` §3.
REML nested random effects (`statsmodels` MixedLM, variance components). **Variance decomposition
is the primary analysis; classification was not run.** No size correction attempted — M4-B's
limitation stands and M4-D is deferred.

## The question

M4-B established that 26–33% of ranked head-shape variation sits between genera. That is a
descriptive first cut and does not say *where in the hierarchy* the signal lives. M4-C asks whether
the genus-level signal is genuinely concentrated at genus, or largely a consequence of species
composition within genera.

## Decomposition

**Model A — genus / species, all admissible specimens**

| trait | genus | species-within-genus | residual |
|---|--:|--:|--:|
| **HW/WL** (primary, n=753) | **23.7%** | 17.7% | 58.6% |
| HL/WL (secondary, n=711) | 25.7% | 21.3% | 53.0% |

**Model B — subfamily / genus / species, subfamily-mapped subset**

| trait | subfamily | genus | species | residual |
|---|--:|--:|--:|--:|
| **HW/WL** (n=544) | 7.7% | **20.2%** | 13.7% | 58.4% |
| HL/WL (n=514) | 8.7% | 23.8% | 16.2% | 51.3% |

**Model B is secondary.** It covers **72%** of each admissible set. Per §3 and the M4-A caveat, the subfamily figure is
**coverage-limited**: it is a lower bound obtained where `GENUS_TO_SUBFAMILY` has a mapping, not an
estimate of the subfamily effect across the whole corpus. That subfamily (7.7%) sits below genus
(20.2%) is informative, but with incomplete mapping **no strong claim is made about the presence or
absence of deep-clade structure** in either direction.

Total taxonomic association: **~41% (HW/WL)** and **~47% (HL/WL)** of variance in validated head
shape is associated with taxonomy at genus level or above/below. Genus is the **largest single
taxonomic component** in every model.

## The robustness check that changes the answer for HL

273 of 464 species are singletons, so species-level variance rests on the 191 replicated species.
Re-fitting Model A on replicated species only:

| trait | | genus | species | residual |
|---|---|--:|--:|--:|
| **HW/WL** | all (n=753) | 23.7% | 17.7% | 58.6% |
| | replicated only (n=483) | **22.5%** | 19.8% | 57.7% |
| HL/WL | all (n=711) | 25.7% | 21.3% | 53.0% |
| | replicated only (n=442) | **21.2%** | **24.1%** | 54.7% |

**The two primary traits answer the question differently, and this is the M4-C result:**

- **HW/WL — genus concentration is ROBUST.** Genus exceeds species-within-genus in both the full
  (23.7 vs 17.7) and the singleton-free (22.5 vs 19.8) fit. The genus-level signal is genuinely a
  genus-level property, not species composition.
- **HL/WL — genus concentration is NOT robust.** The ordering **flips** when singletons are removed
  (21.2 vs 24.1): most of HL's apparent genus signal is attributable to which species happen to sit
  in each genus. HL's decomposition should not be read as evidence of genus-level head-length
  structure.

This is the third independent axis on which HW and HL separate — after M3b/M4-A admissibility
(99.5% vs 93.9%) and the genus-structured HL exclusion. **HW is the trait that carries M4.**

## What the residual is, and what it is not

**M4-C tells you where variation is *associated*, not how much of the residual is pipeline error.**
With one measurement per specimen, the residual (53–59%) combines *within-species biological
variation + unmodelled caste effects + measurement/reconstruction variation*, and these are not
separable in this design. It must not be read as a pipeline-quality figure.

The residual is **not** all measurement error. It contains within-species biological
variation, which in this corpus includes genuine worker polymorphism — *Pheidole* (n=39) and
*Carebara* have major/minor castes with markedly different head proportions, and caste was not
modelled here. Decomposing residual into polymorphism versus pipeline noise would need the caste
labels present in the antscan metadata, and is not attempted: metadata covers only 37% of the
corpus (M4-A §4b).

## Limitations

1. **Size is not controlled.** M4-B's limitation carries: these ratios are scale-free by
   construction, but head proportions scale with body size, and absolute size is unidentifiable on
   `worker_ALT`. The taxonomic variance reported here may partly be size variation expressed as
   shape. This is exactly the estimand M4-D must be reformulated around once scale is resolved.
2. **Subfamily is coverage-limited** (72%), and 13 subfamilies are heavily unbalanced
   (*Myrmicinae* 254 vs *Myrmeciinae* 2).
3. **HL inherits the genus-structured M4-A exclusion** (n=711, permutation p=0.025) on top of the
   singleton sensitivity above.


## Provenance

`statsmodels` 0.15.0 (with `patsy`, `formulaic`, `narwhals`, `wrapt`, `interface-meta`) was
installed into the `pytorch3d` conda env on 2026-09-10 to fit the REML variance components. This is
an **environment change, not a change to the experiment**: the analysis run is the one specified in
§3 of the pre-registration, unmodified. Recorded here so the environment is reproducible.
