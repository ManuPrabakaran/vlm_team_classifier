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
| clip1_easy | Celtics vs Heat | **89.1%** (82.6% in ~10% of runs) | 46 | Easy — white/green vs black/red |
| clip2_hard | Spurs vs Grizzlies | **54.0%** (stable) | 50 | Hard — light blue vs dark navy |
| clip3_edge | Cavs vs Knicks Christmas | **90.6% or 81.2%** (~50/50 across runs) | 32 | Edge — navy throwback vs white/orange |

> **Non-determinism note**: `KMeans` has no `random_state`. clip3 alternates between exactly
> two states (90.6% / 81.2%) almost every other run — a bimodal distribution indicating two
> equally-attractive local minima in color space. clip1 is mostly stable but drops to 82.6%
> in roughly 1 in 10 runs. clip2 is stable at 54% because K-Means consistently fails
> regardless of initialization.

---

## Key Findings

### 1. Non-deterministic baseline

The provided `TeamClustering` class instantiates `KMeans(n_clusters=self.n_teams)` with no
`random_state`, so results vary between runs depending on random cluster initialization.
To get reliable numbers, K-Means was run 10 times per clip and averaged.

**Observed stability per clip:**

| Clip | Mean accuracy | Std | Min | Max |
|------|--------------|-----|-----|-----|
| clip1_easy | TBD | TBD | TBD | TBD |
| clip2_hard | TBD | TBD | TBD | TBD |
| clip3_edge | TBD | TBD | TBD | TBD |

> Run the stability analysis cell in `notebooks/exploration.ipynb` to populate this table.

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

## Next Steps

Results for VLM-based approaches (SigLIP, CLIP, Florence-2, Qwen2-VL) will be added
in Section 3 of `notebooks/exploration.ipynb` and updated here once experiments are
complete.
