# Measuring the Instrument: Perturbation-Artifact Bias in Explanation Evaluation for Cross-Generator Deepfake Detection

**BTP research plan — September 2026**

---

## 0. Executive summary

Your starting instinct was *"nobody has measured explanation faithfulness on a
mixture of generator sources."* That instinct points at a real gap, but as
originally stated it is **partially pre-empted** and rests on one terminology
slip. This document keeps the part that survives, fixes the slip, and adds a
second, stronger contribution that makes the first one defensible.

**The slip.** "Mixture of two or more generators like Vision Transformer and
CNN" conflates two orthogonal axes. ViT and CNN are *detector* architectures.
GAN, diffusion and autoregressive models are *generators*. Both axes are worth
studying and this plan studies both, but they must be named separately or a
reviewer will stop reading at the abstract.

**The pre-emption.** A June 2026 preprint, *The Calibrated Deepfake Trust Score*
(CDTS, arXiv 2606.29484), already reports that Grad-CAM explanation faithfulness
rises and falls with detector competence across 20 generators, on two CNNs and a
CLIP ViT. A plain "faithfulness across generators and architectures" study would
now read as a replication.

**What is still open, and what we claim.** CDTS measured faithfulness with the
standard perturbation machinery and took that machinery at face value. Nobody
has asked whether the machinery is valid for this model class in the first
place. It is not obviously valid, and that is our opening.

> **Core claim.** Perturbation-based faithfulness metrics are systematically
> biased when the model under test is a forgery detector, because the fill
> operator injects exactly the kind of low-level statistical artifact the
> detector was trained to fire on. The measuring instrument emits the signal it
> is trying to measure.

Everything else follows from that claim: once the instrument is corrected, the
cross-generator comparison you wanted to run becomes meaningful, and it can be
run as a *controlled intervention* rather than the correlational study CDTS did.

---

## 1. Positioning against the literature

| Work | What it did | What it left open |
|---|---|---|
| Tsigos et al., MAD'24 @ ICMR (`10.1145/3643491.3660292`) | Compared Grad-CAM++/RISE/SHAP/LIME/SOBOL on one EfficientNetV2 FF++ detector; found LIME most faithful | One CNN, one manipulation family. The fill operator is fixed and never questioned. |
| Tsigos et al., WACVW 2025 (arXiv 2502.03957) | Added adversarially-generated samples to the protocol | Still one CNN. Adversarial perturbation sidesteps the occlusion problem without ever naming or quantifying it. |
| **CDTS, arXiv 2606.29484 (Jun 2026)** | Faithfulness ∝ detector competence, 20 generators, 2 CNNs + CLIP ViT | **Observational, not interventional** — compares detectors that happen to differ in competence. Uses one XAI method (Grad-CAM) and the default occlusion operator, unvalidated. |
| ROAD (Rong et al., ICML 2022); FUD; LILI (arXiv 2607.15482); *Photorealistic Inpainting for Perturbation-based Explanations* (arXiv 2510.03317) | Recognised the out-of-distribution confound in deletion metrics for **natural-image classifiers**; proposed linear imputation and generative inpainting as fixes | Never applied to forensic detectors — where the confound is categorically worse, and where **the standard fix is itself invalid** (see §2.3). |
| XPlainVerse / TriDF / AIGI-Holmes (2025–26) | Natural-language explanation grounding for MLLM detectors | Text-side metrics. Says nothing about pixel-attribution validity, and is a fast-moving, high-pre-emption space we deliberately avoid. |

**Our slot:** the intersection nobody occupies — *XAI-evaluation methodology*
(slow-moving, low pre-emption) × *forensic detectors* (where the standard
methodology breaks) × *cross-generator shift* (your original idea, now
executable on a valid instrument).

---

## 2. The three contributions

### C1 — Perturbation leakage: a null test for forensic XAI *(primary, methodological)*

Every deletion/insertion/ROAD/pixel-flipping metric has the same shape: remove
the pixels an explanation calls important, watch the prediction move. "Remove"
is not defined by the metric — it is defined by an **operator** that decides
what goes in the hole. Fill with black, grey, blur or noise.

For an ImageNet classifier, that choice is a second-order detail. For a forgery
detector it is first-order, because a forgery detector *is* an artifact
detector. A hard mask boundary is a splice edge. A flat grey region has
impossible sensor statistics. White noise has a flat power spectrum no camera
produces. Each is a textbook forgery cue.

**The test needs no explanation method at all:**

> Take images that are genuinely **real**. Mask a **random** region — there is
> no saliency map involved. Measure how far P(fake) moves.

Under an unbiased operator that is ~0: scrambling arbitrary pixels of a genuine
photograph should not manufacture evidence of forgery. Whatever it actually is,
is the operator's **leakage**, and it contaminates every faithfulness number
ever computed with that operator.

Three reported quantities (implemented in `src/xgf/metrics/leakage.py`):

- `leakage_auc` — area under P(fake) vs masked fraction, on real images
- `leakage_slope` — dP(fake)/d(fraction) near the origin, the regime deletion curves actually read
- `flip_rate` — fraction of real images pushed across the decision threshold by masking alone

Reported per (detector, operator, generator-family). **The family axis is the
bridge to C2:** if leakage differs across families, then cross-generator
faithfulness comparisons made with a leaky operator are not comparing like with
like — which is precisely the confound sitting under your original idea.

### C2 — Cross-generator faithfulness, as a controlled intervention *(your idea, sharpened)*

Same backbone, same head capacity, same training budget. The **only** thing that
varies is the generator diversity of the training set:

- one probe per single family (`gan`, `diffusion`, `inpainting`, …)
- one probe on the **mixture** of all families
- leave-one-family-out probes

Then evaluate every probe on every family, and report **two** matrices:

1. detection AUC (the accuracy matrix — known territory), and
2. leakage-corrected faithfulness ΔAUC (the faithfulness matrix — new).

Three questions fall out:

- **Q2a** Does faithfulness degrade under generator shift *faster or slower*
  than accuracy? A detector that keeps its AUC on an unseen generator while its
  explanation collapses is right for the wrong reason — and that is exactly the
  failure mode a forensic analyst must not be handed.
- **Q2b** Does training-set generator **diversity** buy faithfulness robustness
  over and above the accuracy robustness it is already known to buy? This is the
  causal version of CDTS's correlational finding.

  **Design note, learned the hard way:** the `mixed` probe *cannot* answer this.
  It trains on every family, so it has no unseen family and no generalization
  gap by construction — the first pilot run returned `NaN` for exactly this
  reason. The contrast that works is **single-family vs leave-one-out on the
  same held-out family**: family C is scored once by a probe trained on A alone
  (diversity 1) and once by a probe trained on {A,B} (diversity 2). Same head
  capacity, same test images, only training diversity differs. `mixed` is kept
  purely as an upper-bound reference point.
- **Q2c** Does the CNN-vs-ViT-vs-CLIP architecture axis interact with the
  generator axis? (Your "ViT vs CNN" interest, correctly named.)

Because C1 comes first, the faithfulness numbers in C2 are measured on an
instrument whose bias has been quantified and subtracted — which is what makes
C2 more than a replication of CDTS.

### C3 — Artifact-neutral perturbation operators *(the fix)*

We implement and compare operators across the bias spectrum
(`src/xgf/perturb/operators.py`):

*Artifact-injecting (the field standard):* `zero`, `gray`, `mean`,
`uniform_noise`, `gaussian_blur`

*Artifact-neutral candidates (ours):*
- `road_linear` — ROAD's noisy linear imputation from surviving neighbours
- `spectral_match` — noise shaped to the image's **own** power spectrum, so the
  fill removes local content without handing the detector an alien spectral
  signature. Motivated specifically by the fact that many synthetic-image
  detectors read frequency fingerprints.
- `real_transplant` — paste co-located pixels from an authentic photograph, so
  the fill carries genuine sensor/demosaicing/JPEG statistics

**Deliberately excluded, and this exclusion is itself a contribution:**
generative (diffusion) inpainting. It is the fashionable fix for the OOD
confound in natural-image XAI (LILI, arXiv 2607.15482). It is *invalid here* —
the filler is a generator, so it stamps the hole with precisely the synthesis
artifacts under investigation. Using a diffusion model to evaluate a
diffusion-image detector is measuring a ruler with itself. We can demonstrate
this empirically by including a diffusion-inpaint arm and showing it has the
**worst** leakage of all.

**The headline result to aim for:** run the Tsigos et al. comparison under both
a leaky and a neutral operator. If the ranking of explanation methods changes —
if "LIME is the most faithful" does not survive — then a published conclusion
about explanation methods was partly a statement about the fill operator. The
codebase computes that rank correlation automatically.

---

## 2.5 Preliminary validation (synthetic data — **not** a result)

The pipeline was run end-to-end on the synthetic self-test set (ResNet50 frozen
+ mixed-family probe, 20 real images, 2 repeats). These numbers exist only to
show the machinery detects what it is supposed to detect; the "generators" are
image corruptions, not real models.

| operator | class *(hypothesis)* | leakage ΔAUC |
|---|---|---|
| `zero` | injecting | **0.312** |
| `road_linear` | neutral | **0.308** |
| `gaussian_blur` | injecting | 0.304 |
| `uniform_noise` | injecting | 0.262 |
| `gray` | injecting | 0.252 |
| `mean` | injecting | 0.250 |
| `real_transplant` | neutral | **0.131** |
| `spectral_match` | neutral | **0.114** |

Three things to notice, all of which the real experiment must re-test:

1. **The effect is large.** Masking random regions of images that are *genuinely
   real* drives P(fake) from a 0.43 baseline to 0.75–0.81 under injecting
   operators. On a toy detector, at least, the instrument manufactures more
   signal than most explanations could hope to.
2. **The predicted separation appears.** `spectral_match` and `real_transplant`
   leak ~2.5× less than the injecting group, and — a useful sanity check — their
   curves *return to baseline* at 100 % masking, because a fully transplanted
   image really is a real image. `zero` instead ends at 0.81: an all-black image
   is confidently "fake".
3. **`road_linear` behaves like an injecting operator.** ROAD's noisy linear
   imputation is the strongest published fix for the OOD confound in
   natural-image XAI, and here it leaks as badly as black fill. The likely
   reason is that linear imputation produces smooth, texture-free regions, and
   absent high-frequency content is itself a forgery cue. If this survives on
   real data it is a genuinely interesting negative result: **the standard
   correction from natural-image XAI does not transfer to forensics**, which
   strengthens the case for C3's forensics-specific operators.

Caveat worth stating in the paper: on this toy setup `flip_rate` is 1.0 for
every operator because the probe's baseline P(fake) on real images is 0.43 —
too close to the 0.5 threshold for the flip rate to discriminate. On real data
with a well-separated detector it will.

---

## 3. Why this is BTP-sized and low-risk

- **Inference-only.** No backbone is ever trained. Every detector is a frozen
  backbone plus a logistic-regression head fitted on cached features — seconds
  of CPU, not GPU-days.
- **Unambiguous metrics.** Leakage AUC, ΔAUC, Spearman rank agreement. No
  human study, no annotation, no IRB.
- **Negative results are publishable.** If leakage turns out small, "the
  standard protocol is valid for forensic detectors, and here is the null test
  that establishes it" is a useful, citable methodological result. The project
  cannot fail to produce a finding.
- **Low pre-emption.** XAI-evaluation methodology moves far slower than the
  explainable-MLLM space. The 2024 Tsigos paper is still the reference point.

**Main risks and mitigations**

| Risk | Mitigation |
|---|---|
| Leakage turns out negligible | That *is* the result — it validates the protocol and still leaves C2 intact on a now-justified instrument |
| CDTS authors extend to operators first | Our angle is the operator itself, not competence; and C3's neutral operators are a positive contribution they have not signalled |
| Hardware (4 GB VRAM) | Frozen ViT-B/16 inference fits in ~1.5 GB fp16; see §5 |
| Datasets gated behind request forms | ArtiFact, ForenSynths, CocoGlide are all downloadable today with no form; FF++/DF40 requested day 1 as a bonus axis |

---

## 3.5 Known limitations — state these before a reviewer does

**The random-control subtraction is a partial correction, not a cure.**
`ΔAUC = AUC_random − AUC_salient` removes the operator's *average* effect at a
given masked area. It does not remove any **interaction** between the operator
and the specific regions an explanation selects. If salient regions are
systematically smoother, higher-frequency or nearer the image border than random
ones, the operator will behave differently on them and some bias survives the
subtraction. Mitigations, all cheap: match the control's mask geometry to the
salient mask (already done — controls use spatially-coherent blobs, not
scattered pixels), and report results under several operators so a conclusion
that only holds for one is visible as such. Fully removing the interaction term
would need a matched-region control, which is a good extension but not required
for the main claim.

**`spectral_match` randomises phase globally, not locally.** It preserves the
image's global power spectrum but discards spatial structure inside the hole.
For a detector keyed on *local* spectral anomalies this is less neutral than it
looks. A patch-local variant is the obvious refinement if the numbers warrant it.

**`real_transplant` still leaves a seam.** The fill has authentic statistics but
the mask boundary is a real discontinuity. Every non-generative operator shares
this, which is exactly why the null test measures it rather than assuming it away.

**Leakage is a property of a (detector, operator) pair, not of an operator.**
A detector that ignores high frequencies will not care about `uniform_noise`.
So the leakage table must be reported per detector, and claims must be phrased
as "for detectors of this class" rather than universally.

**Synthetic self-test data proves nothing scientific.** It exists to debug the
pipeline. Every number in §2.5 must be regenerated on ArtiFact / ForenSynths /
CocoGlide before it appears in a paper.

**Large negative ΔAUC is a warning sign, not a finding.** The first pilot run
produced significantly *negative* ΔAUC for Grad-CAM on the seen family
(d = −8.4, p = 2×10⁻⁶) — i.e. deleting the "most important" pixels held P(fake)
*higher* than deleting random ones. Anti-faithful saliency is a real phenomenon
in the literature, but an effect that large on toy data more likely reflects the
operator×region interaction described above: Grad-CAM's chosen regions happen to
produce less artifact under `zero` fill than random regions do. Treat any large
negative ΔAUC as a prompt to check the operator, not as a result about the
explanation method.

---

## 4. Experimental plan

| Phase | Weeks | Work | Go/no-go |
|---|---|---|---|
| P0 Setup | 1 | CUDA torch, ArtiFact + ForenSynths + CocoGlide downloaded, manifest built, pipeline validated on synthetic self-test data | `scripts/05` runs end-to-end |
| P1 Detectors | 2 | Extract features for ResNet50 / ViT-B/16 / CLIP-ViT-B/16; train all probe configs; accuracy matrix | Mixed probe beats single-family probes on unseen families |
| P2 **C1 leakage** | 3–4 | Full operator × detector × family leakage sweep | Injecting vs neutral operators separate significantly |
| P3 **C3 rankings** | 5–6 | Explanation-method rankings under each operator; rank-agreement tests | Either rankings diverge (strong result) or they do not (protocol-validation result) |
| P4 **C2 cross-generator** | 7–9 | Faithfulness transfer matrix; accuracy gap vs faithfulness gap | Both matrices complete with error bars |
| P5 Write-up | 10–12 | Paper, ablations (JPEG/resize robustness), public code release | Camera-ready draft |

**Publication targets** (ordered by fit): ACM MAD workshop @ ICMR — the
baseline's home venue and the natural place to say "your protocol has a bug";
IEEE WIFS / ICIP / IJCB; CVPR/ICCV/WACV workshops (DFAD, WMF, AI4MFDD — WACVW
hosted the Tsigos follow-up); ACM IH&MMSec.

---

## 5. Compute: do you need a GPU?

Measured on your machine (16-core CPU / 12 torch threads, 4 GB laptop GPU),
ResNet50 @224px: **7–8.6 images/s on CPU** (7.1 measured under load, 8.6 idle).
Reproduce with `python scripts/00_setup_check.py --benchmark`.

**Training: no GPU needed, and barely any compute.** Backbones are frozen;
heads are logistic regression on cached features. The full set of probe configs
trains in **seconds**. This is the single most important design decision in the
project and it is what makes the 4 GB card a non-issue for training.

**Evaluation: GPU strongly recommended.** The cost is forward passes for the
perturbation curves, and it dominates everything:

Full budget (500 eval images/cell, 25 mask steps, RISE at 2000 masks,
3 backbones):

| Experiment | Forward passes | CPU @ 8.6 img/s | GPU @ ~120 img/s |
|---|---|---|---|
| C1 leakage sweep (3 detectors × 8 ops) | ~0.6 M | ~18 h | ~1.5 h |
| Feature extraction (3 backbones, 60 k imgs) | 180 k | ~6 h | ~25 min |
| C2 faithfulness grid, cheap explainers | ~10.8 M | ~15 days | ~25 h |
| C3 RISE + LIME arm | ~18 M | ~24 days | ~42 h |
| **total** | ~29.6 M | **~40 days** | **~3 days** |

**Verdict: get the GPU working.** Full budget on CPU is ~40 days — not viable.
On your 4 GB card it is **~3 days**, which is fine for a BTP.

If the GPU truly cannot be used, the project is still deliverable on CPU by
shrinking three knobs (all exposed as CLI flags, and pre-set in
`configs/cpu_budget.yaml`):

- drop RISE and LIME — keeps Grad-CAM++/Grad-CAM/IG/attention-rollout/random (removes 60 % of the total)
- `--n-images 500 → 150` (3.3× off everything that remains)
- `--n-steps 25 → 15` (1.7×)

That lands at roughly **3 days of CPU** for a complete, defensible result set.
Note what is lost: LIME is the method Tsigos et al. crowned most faithful, so
dropping it weakens C3's headline re-test. If CPU is the only option, prefer to
keep LIME on a *reduced image count* rather than cutting it entirely.

**C1 alone — the primary contribution — costs ~18 h on CPU**, so even in the
worst case the main claim is reachable in a weekend.

**Two things to fix on your machine first** (both flagged by
`scripts/00_setup_check.py`):

1. `torch 2.2.2+cpu` is installed — it cannot see your GPU at all. Update the
   NVIDIA driver (currently 516.40), then
   `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`.
2. 4 GB VRAM rules out CLIP ViT-L/14 at comfortable batch sizes. Use
   **ViT-B/16** for all three arms (ResNet50, ViT-B/16, CLIP-ViT-B/16) at fp16,
   batch 16–32. That fits in ~1.5–2 GB and keeps the architecture comparison
   fair, since all three are similar capacity.

---

## 6. Datasets

Chosen so work can start today without waiting on any access form.

| Dataset | Families | Masks | Access | Role |
|---|---|---|---|---|
| **ArtiFact** (Kaggle `awsaf49/artifact-dataset`, 32 GB) | 13 GAN + 7 diffusion + 5 other, 2.5 M images | no | open | **Primary.** Per-generator folders make the transfer matrix trivial |
| **ForenSynths** (CNNDetection, Wang CVPR'20) | 8 GAN subsets | no | open | GAN arm, standard baseline |
| **CocoGlide** (GRIP-UNINA) | GLIDE inpainting, 512 imgs | **yes** | open | Localization/plausibility arm |
| **GenImage** (HuggingFace) | 8 diffusion generators | no | open | Diffusion arm at scale |
| FF++ / DF40 | faceswap, reenactment | yes | request form | Bonus face axis; request day 1 |

Ground-truth masks matter because they separate two things this field routinely
conflates: **faithfulness** (does the map reflect what the *model* used — no
ground truth needed) from **plausibility** (does it point at what was *actually*
manipulated — needs a mask). A detector can be perfectly faithful and completely
implausible: the map honestly reports that the model keyed on a background JPEG
grid.

---

## 7. What to do first

```bash
python scripts/00_setup_check.py --benchmark     # measure YOUR throughput
python scripts/01_build_manifest.py --selftest   # synthetic data, no download
python scripts/02_extract_features.py --manifest manifest_selftest.csv --backbone resnet50
python scripts/03_train_probes.py   --backbone resnet50 --tag selftest
python scripts/04_run_leakage.py    --backbone resnet50 --tag selftest   # <- C1
python scripts/05_run_faithfulness.py --backbone resnet50 --tag selftest # <- C2/C3
```

The self-test path generates synthetic images with deliberately distinct
frequency signatures per "family". It is not a scientific artifact — it exists
so the entire pipeline, every metric and every figure can be debugged while the
32 GB ArtiFact download runs.
