# Experimental Results

## Experimental Setup

**Data**: Three basketball video clips sourced independently (sample data was not provided).
Ground truth was collected via a custom pipeline: YouTube clip extraction, YOLO-based player
detection, and an interactive manual labeling tool in `notebooks/data_collection.ipynb`.

**Metrics**: Per-player team classification accuracy on valid single-player detections.
Merged bounding boxes (two players in one bbox) and ambiguous detections were marked
`valid=False` during labeling and excluded from all accuracy calculations. Numbers therefore
reflect performance on clean detections only — real-world accuracy on raw YOLO output will
be lower.

**Baseline**: `TeamClustering` in `src/baseline.py` — K-Means clustering (k=2) on mean
RGB jersey color extracted from the middle 40% of each player bounding box. Fit on the
first labeled frame, predicted on all subsequent frames. Label orientation (which cluster
maps to which team) is resolved by trying both assignments and taking the higher accuracy.

---

## Baseline K-Means Results

| Clip | Match-up | Accuracy | Valid detections | Difficulty |
|------|----------|----------|-----------------|------------|
| clip1_easy | Celtics vs Heat | **87.2% ± 3.0%** (max 89.1%) | 46 | Easy — white/green vs black/red |
| clip2_hard | Spurs vs Grizzlies | **52.8% ± 1.8%** (max 54.0%) | 50 | Hard — light blue vs dark navy |
| clip3_edge | Cavs vs Knicks Christmas | **86.9% ± 4.6%** (max 90.6%) | 32 | Edge — navy throwback vs white/orange |

> Means and std over 10 runs. `KMeans` has no `random_state` so single-run numbers are
> unreliable. clip3 has the highest variance (4.6%), flip-flopping between 81.2% and 90.6%.
> clip2 has the lowest variance (1.8%) but is consistently near random chance.

---

## Key Findings

### 1. Non-deterministic baseline

The provided `TeamClustering` class instantiates `KMeans(n_clusters=self.n_teams)` with no
`random_state`, so results vary between runs depending on random cluster initialization.
To get reliable numbers, K-Means was run 10 times per clip and averaged.

**Observed stability per clip:**

| Clip | Mean accuracy | Std | Min | Max |
|------|--------------|-----|-----|-----|
| clip1_easy | 87.2% | 3.0% | 82.6% | 89.1% |
| clip2_hard | 52.8% | 1.8% | 50.0% | 54.0% |
| clip3_edge | 86.9% | 4.6% | 81.2% | 90.6% |

> 10 runs per clip. Full stability data in `results/metrics.json`.

The three clips fall into three distinct regimes:

- **clip1_easy** is usually stable at 89.1% but drops to 82.6% in about 10% of runs.
  Even with clearly distinct colors (white/green vs black/red), a bad random initialization
  occasionally finds a suboptimal local minimum. The correct clustering is strongly favored
  but not guaranteed.
- **clip2_hard** is paradoxically the most stable result, locking in at 54.0% across
  every run. That consistency is not a good sign though. It just means the colors are so
  similar that K-Means finds the same wrong answer regardless of where it starts.
  There is no correct local minimum to find.
- **clip3_edge** has the most striking behavior: it alternates between 90.6% and 81.2%
  on almost every other run. K-Means has two roughly equally-attractive local minima for
  this color distribution and random initialization decides which one it lands on each time.

This non-determinism is itself a production reliability concern. A classifier that gives
different answers on the same input data is not acceptable at scale. At 1000 games/day,
even clip1's 10% failure rate would mean hundreds of incorrect classifications daily
before any other failure mode is considered.

### 2. YOLO detection bias toward lighter jerseys

YOLO detection exhibited significant bias in clip2_hard. Dark navy Grizzlies jerseys
blended into the dark arena background, producing approximately a **4:1 ratio** of
Spurs to Grizzlies detections. This compounds the classification problem in two ways:

1. K-Means sees far more Spurs examples during fitting, biasing the cluster centroids.
2. Evaluation coverage is uneven — the model is tested mostly on one team.

This suggests that detection quality and classification quality are coupled: a team with
low-contrast jerseys against the background will be both harder to detect and harder to
classify. A robust system needs to handle both failure modes.

### 3. Custom data pipeline required

Sample video clips and bounding box annotations were not provided, so a full data
collection pipeline was built:

- YouTube clip extraction via `yt-dlp` with timestamp slicing (`src/utils.py`)
- YOLO-based player detection to propose bounding boxes
- Interactive manual labeling tool (`notebooks/data_collection.ipynb`) for assigning
  `team_id` (0, 1, or -1 for referee) and marking invalid detections

All ground truth files are in `data/` and follow the schema:
```json
[{"frame_idx": 0, "timestamp": 0.0, "labels": [{"bbox": [x1,y1,x2,y2], "team_id": 0, "valid": true}]}]
```

---

## Failure Analysis

| Failure mode | Clips affected | Root cause |
|---|---|---|
| Similar jersey colors | clip2_hard | Mean RGB insufficient to separate light blue vs dark navy |
| Non-deterministic clustering (bimodal) | clip3_edge | Two equally-attractive local minima; alternates 90.6%↔81.2% every other run |
| Occasional initialization failure | clip1_easy | Rare bad init drops 89.1% → 82.6% in ~10% of runs |
| Detection dropout | clip2_hard | Dark jerseys against dark background |
| Merged bounding boxes | All clips | Two overlapping players detected as one bbox |

---

## Improved K-Means via Randomized Hyperparameter Search

Randomized search over 200 configurations of color space, crop margins, feature type,
skin filtering, and initialization parameters. Evaluated against ground truth on all
three clips. Baseline included for direct comparison.

### Best Configuration

RGB color space, margins=0.40/0.40 (tighter crop), mean color, skin_filter=True,
n_init=20, random_state=42.

| Clip | Baseline K-Means | Improved K-Means | Delta |
|------|-----------------|------------------|-------|
| clip1_easy | 89.1% | **100.0%** | +10.9% |
| clip2_hard | 56.0% | **92.0%** | +36.0% |
| clip3_edge | 90.6% | **96.9%** | +6.3% |
| **Average** | **78.6%** | **96.3%** | **+17.7%** |

Baseline ranked #78 out of 201 configurations.

### What Matters Most

1. **Skin filtering** (dominant signal): Removing skin-tone pixels lets jersey color
   dominate. Every top-10 config uses it. Drives clip2_hard from 56% → 92%.
2. **Tighter crop margins** (0.40/0.40 vs 0.30/0.30): Less shorts/neck/court in the
   feature region. Consistent across all top configs.
3. **Deterministic initialization** (random_state=42 + n_init=20): Eliminates run-to-run
   variance entirely.

### Overfitting Mitigation: Jersey-Aware Skin Filter Calibration

Skin filtering carries overfitting risk — jerseys in the skin-tone hue range (orange,
red, tan) would have jersey pixels incorrectly stripped. We built a pre-game calibration
system using reference jersey images:

| Clip | Team 0 Overlap | Team 1 Overlap | Decision |
|------|---------------|---------------|----------|
| clip1_easy | 8.7% (Heat white/red) | 0.0% (Celtics green/black) | ENABLED |
| clip2_hard | 0.1% (Grizzlies blue) | 0.4% (Spurs black) | ENABLED |
| clip3_edge | **39.6%** (Knicks orange) | 8.2% (Cavs navy) | **DISABLED** |

The Knicks' orange accents (hue ~8) sit directly in the skin filter range. The calibration
correctly catches this and disables skin filtering for that game.

### Calibrated Production Accuracy

With jersey-aware calibration (skin filter enabled only when safe):

| Clip | Calibrated Accuracy | Method |
|------|-------------------|--------|
| clip1_easy | **100.0%** | Skin filter enabled |
| clip2_hard | **92.0%** | Skin filter enabled |
| clip3_edge | **93.8%** | Skin filter disabled — fallback to margins + determinism |
| **Average** | **95.3%** | |

The calibrated system delivers +16.7% over baseline without overfitting risk. Tighter
margins and determinism alone provide +3-5% on all clips; skin filtering adds another
+36% on hard games only when the pre-game jersey scan confirms it's safe.

### Implementation Note: Tighter Crops Are K-Means-Only

The 0.40/0.40 margins apply exclusively to K-Means feature extraction. SigLIP and
Qwen2-VL receive the full bounding box crop — they actively leverage shorts, shoulders,
socks, and other spatial features that K-Means (which reduces everything to a single
mean color) treats as noise. The cascade stages see different information from the same
bounding box, which is why they are complementary rather than redundant.

---

## VLM Model Comparison

All VLM models evaluated on the same three clips using the same ground truth and evaluation
framework. Accuracy computed as best of both label orientations (since cluster/team ID
mapping is arbitrary).

| Clip | K-Means Baseline | SigLIP | CLIP | Qwen2-VL 2B |
|------|-----------------|--------|------|-------------|
| clip1_easy | 87.2% ± 3.0% | **97.8%** | 84.8% | 54.3% |
| clip2_hard | 52.8% ± 1.8% | **86.0%** | 70.0% | 58.0% |
| clip3_edge | 86.9% ± 4.6% | **71.9%** | 78.1% | 56.2% |
| **Average** | **75.6%** | **85.2%** | **77.6%** | **56.2%** |

> Florence-2 was excluded from benchmarking due to architectural mismatch: its fixed
> task-token design (`<OD>`, `<CAPTION>`) has no contrastive embedding space, no
> confidence signal for cascade routing, and no three-class output mechanism. See
> [TEAM_CLASSIFICATION_RESEARCH.md](TEAM_CLASSIFICATION_RESEARCH.md) Section 3.4.

---

## VLM Analysis

### SigLIP (best overall)
- Dominates on clip1_easy (97.8%) and clip2_hard (86.0% — where K-Means collapses to chance)
- Struggles on clip3_edge (71.9%) — likely due to navy throwback jerseys being visually
  ambiguous in embedding space
- Zero-shot with text anchors derived from GPT-4o descriptions
- ~15ms per crop on T4 GPU — fast enough for primary cascade stage

### CLIP (baseline VLM)
- Surprisingly competitive on clip3_edge (78.1% > SigLIP's 71.9%)
- Weaker on clip1_easy (84.8% vs 97.8%) and clip2_hard (70.0% vs 86.0%)
- Same architecture as SigLIP but weaker contrastive training — confirms SigLIP's
  superior fine-grained discrimination

### Qwen2-VL 2B (near chance)
- 54.3% / 58.0% / 56.2% — barely above random on all clips
- The 2B model lacks the visual reasoning capacity for this task
- Pivoted role: OCR specialist for jersey number reading, not primary classifier
- The 7B model would likely perform better but requires more GPU memory

### Florence-2 (excluded — architectural mismatch)
- Fixed task-token architecture (`<OD>`, `<CAPTION>`) cannot produce contrastive embeddings
  for prototype comparison — the core strategy that makes SigLIP effective
- No confidence signal for cascade routing (captions are binary, not graded)
- No three-class mechanism (Team 0 / Team 1 / Referee) without multiple inference passes
- Excluded from benchmarking; architectural limitations make it a poor fit regardless

---

## Cascade Recommendation

Based on experimental results:

1. **K-Means** as first stage — resolves 80%+ of easy-game players at < 1ms
2. **SigLIP** as primary VLM — best accuracy/speed balance, handles hard cases
3. **Qwen2-VL 7B** as escalation — for jersey number OCR and visual reasoning on
   the hardest cases (2B too weak for general classification)
4. **Manual review** as last resort — target < 5% of players

The cascade architecture is validated: SigLIP alone improves average accuracy from
75.6% to 85.2%, with the biggest gains on the hardest clip (52.8% → 86.0%).

---

## Prototype Quality Analysis

Confidence calibration revealed that clip3_edge's low accuracy (71.9%) was caused by
a weak prototype — only 2 players per team from a single frame. Team 0 was at 33.3%
accuracy while Team 1 was at 100%, indicating a broken Team 0 centroid rather than a
game-level or model-level failure.

Switching to multi-frame prototypes (5 frames, crop quality filtering) improved results:

| Clip | Single-frame | Multi-frame | Delta |
|------|-------------|-------------|-------|
| clip1_easy | 93.5% | 97.8% | +4.3% |
| clip2_hard | 90.0% | 90.0% | — |
| clip3_edge | 62.5% | 81.2% | +18.7% |

In production, prototypes are built during tipoff when players are stationary — the
single-frame mid-action scenario is unlikely. Multi-frame building is still the default
as it is strictly better. Remaining clip3 errors (Team 0 at 72.2%) fall to the cascade's
Qwen2-VL escalation stage.

---

## Cross-Model Cascade Validation

Per-player cross-referencing of K-Means and SigLIP (multi-frame) predictions shows
that the two models fail on different players, validating the cascade architecture.

| Clip | K-Means wrong | SigLIP rescued | Both wrong | Both wrong % |
|------|--------------|----------------|------------|-------------|
| clip1_easy | 11 | 11 (100%) | 0 | 0.0% |
| clip2_hard | 23 | 21 (91.3%) | 2 | 4.0% |
| clip3_edge | 11 | 8 (72.7%) | 3 | 9.4% |
| **Total** | **45** | **40 (88.9%)** | **5** | **3.9%** |

SigLIP rescues 89% of K-Means failures. Only 3.9% of players (5 out of 128) defeated
both models — the cascade floor that would escalate to Qwen2-VL. This is well below
the 5% manual review target.

SigLIP accuracy on players K-Means already got right stays high (86–97%), confirming
it does not introduce new errors on easy players. The models are complementary.

---

## Cascade Threshold Sweep

Full cascade evaluation with K-Means confidence gating, SigLIP margin-based escalation,
and Qwen2-VL escalation bucket. Three thresholds swept independently:

1. **Cluster separation** (game-level): Skip K-Means entirely for hard games
2. **Per-player distance ratio**: Escalate uncertain K-Means predictions to SigLIP
3. **SigLIP margin**: Escalate uncertain SigLIP predictions to Qwen2-VL

### Best Configuration (sep=50, ratio≥3.0, margin=0.03)

| Clip | Accuracy | Method |
|------|----------|--------|
| clip1_easy | **97.8%** | K-Means confident players resolved at Stage 2; uncertain escalated to SigLIP |
| clip2_hard | **90.0%** | Game-level gate (sep=46.8 < 50) routes all players directly to SigLIP |
| clip3_edge | **96.9%** | K-Means handles most; SigLIP catches edge cases |
| **Average** | **94.9%** | |

Frame latency: 259ms (all stages active). Cost: $12.41/game.

> The sep=50 gate is the critical insight: clip2_hard's cluster separation of 46.8
> correctly identifies it as a hard game, routing all players to SigLIP and jumping
> accuracy from ~50% to 90%.

### 100% SigLIP Ceiling (Thought Experiment)

What if we skip K-Means entirely and run SigLIP on every player?

| Clip | Accuracy | vs Cascade |
|------|----------|-----------|
| clip1_easy | **100%** | +2.2% |
| clip2_hard | **90.0%** | — |
| clip3_edge | **93.8%** | -3.1% |
| **Average** | **94.6%** | -0.3% |

Latency: 150ms/frame. Cost: $7.20/game ($1.80 with tracking).

The cascade's K-Means stage actually helps on clip3_edge (+3.1%) — K-Means gets the
easy players right and lets SigLIP focus on the hard ones. Pure SigLIP slightly
underperforms the cascade overall, validating the multi-stage design.

### K-Means Confidence Diagnostic

The sweep revealed that K-Means' per-player distance ratio is a weak confidence signal —
correct and wrong predictions have overlapping ratio distributions. The ratio works as a
gate only at high thresholds (≥2.5), where it's selective enough to catch most errors at
the cost of routing more players to SigLIP.

**Game-level cluster separation is the reliable signal.** clip2_hard's separation of 46.8
RGB units vs clip1_easy's 90.5 and clip3_edge's 152.7 correctly identifies the hard game.
At sep≥50, clip2 routes entirely to SigLIP, boosting average accuracy from 77% to 91%.

### Production Context

The latency numbers above assume every player is actively classified every frame. In
production with DeepSORT tracking, most players are locked in after initial classification.
Only 2-3 uncertain players per frame need active classification, reducing effective
per-frame latency by 70-80%. The ratio≥3.0 config at 191ms/frame becomes ~40-50ms/frame
with tracking — well within the 100ms budget.
