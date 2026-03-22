# Production Plan

## 1. Deployment Architecture

### Hardware
- **GPU**: NVIDIA H100 80GB — sufficient to hold SigLIP (900MB) + Qwen2-VL 2B (8GB) simultaneously with room for batching
- **Target throughput**: 1000+ games/day, 30 FPS per game, <100ms per frame end-to-end

### Model Serving
- **SigLIP**: Always loaded, compiled with TensorRT for 2-3x inference speedup. Handles 80%+ of classifications at ~15ms/crop
- **Qwen2-VL 2B**: Loaded on-demand when game difficulty is high or SigLIP confidence is low. Served via vLLM for efficient batched generation
- **K-Means**: CPU-only, negligible cost — runs on every frame as the first cascade stage

### Service Layout
```
┌──────────────────────────────────────────────────────┐
│  Pre-game Service (CPU)                              │
│  GPT-4o descriptions + roster lookup                 │
│  + referee prototype (fixed constant, loaded once)   │
└──────────────┬───────────────────────────────────────┘
               │ game context JSON
┌──────────────▼───────────────────────────────────────┐
│  Frame Processing Service (GPU)                      │
│  YOLO → Referee Filter → Cascade (K-Means → SigLIP  │
│  → Qwen2-VL) → DeepSORT tracking → Output           │
└──────────────────────────────────────────────────────┘
```

## 2. Pre-game Setup Pipeline

Run once per game before tipoff (~30 seconds, $0.04 in API cost):

1. **Roster lookup**: Fetch team rosters from public API. Flag unique jersey numbers (numbers appearing on only one team) — these allow instant classification when readable
2. **GPT-4o description generation**: Send team uniform images to GPT-4o with contrastive prompting. Generate three description sets:
   - SigLIP text anchors (concise, visual-feature-focused)
   - Qwen2-VL 2B prompts (simple, direct language for the smaller model)
   - Qwen2-VL 7B prompts (detailed, multi-feature descriptions for the larger model)
3. **Game difficulty scoring**: Compute from jersey color similarity, special jersey flags, arena lighting conditions. Stored as a float in [0, 1] — adjusts cascade thresholds for the entire game
4. **Skin filter calibration**: Extract dominant hues from each team's jersey reference images. Compute overlap ratio with skin-tone HSV range (hue 0–20). If either team's jersey has >10% overlap (e.g., orange, red, tan jerseys), disable or narrow the skin filter for K-Means feature extraction. This prevents the improved K-Means from stripping jersey pixels on games where jersey colors fall in the skin-tone range — a validated overfitting risk caught during prototype evaluation (Knicks orange: 39.6% overlap)
5. **Referee prototype loading**: Load the pre-computed SigLIP referee embedding centroid — a fixed constant built once from reference images of NBA referee uniforms (grey shirt, black vertical stripes, black pants) captured from multiple angles. This prototype never changes between games because NBA referee uniforms are standardized across the entire league. Zero per-game cost; the prototype is a static asset loaded from disk
6. **Output**: Game context JSON containing descriptions, roster, difficulty score, skin filter config, referee prototype, and threshold overrides

**Cost at scale**: 1000 games/day × $0.04 = $40/day for description generation. Negligible compared to GPU compute.

## 3. Runtime Pipeline

Per-frame processing at 30 FPS:

### Stage 0: Detection
- YOLO v8 detects all players → bounding boxes
- Filter by confidence > 0.5, person class only
- Typical output: 10 players + 2-3 referees per frame

### Stage 0.5: Referee Filter (near-zero cost)
NBA referee uniforms are **identical across every game** — the same grey shirt with black vertical stripes, black pants. Unlike team jerseys that change per game, referee uniforms are a universal constant. This transforms referee detection from a per-game classification problem into a one-time known-template matching problem.

- **Court position pre-filter**: Rank all detections by x-coordinate. Referees work the sidelines and baselines — flag detections near court edges as referee candidates. This eliminates most team players from consideration before any computation
- **SigLIP prototype comparison**: For candidates that pass the position filter, compute cosine similarity against the fixed referee prototype (a single dot product on a 768-dim vector — microseconds). The SigLIP embedding is already being extracted for team classification downstream, so this adds negligible cost
- **Bounded search**: At most 3 referees on court (usually 2). Once 2-3 matches are found, stop — no more false positives possible
- **Output**: Referee detections receive `team_id = -1` and are **excluded from the entire classification cascade**. This saves 2-3 crops worth of K-Means/SigLIP/Qwen compute per frame and prevents referee jersey colors from polluting team cluster centroids

The referee prototype is built once from ~20 reference images of NBA referees from multiple angles (front, back, side, motion) and stored as a static constant. It never needs updating between games, seasons, or arenas.

### Stage 1: Pre-screening (near-zero cost)
- **Cluster separation check**: If K-Means centroids too close (< 30 RGB units), force VLM escalation for all players
- **Court position heuristic**: Rank remaining team players by x-coordinate. Players on the far left relative to others are either defensive for the left team or offensive for the right. Down-weighted during transitions, free throws, jump balls — returns a prior probability, not a hard assignment
- **Temporal lock**: Players with N+ frames of consistent high-confidence classification skip the cascade entirely

### Stage 2: K-Means (< 1ms/crop)
- Extract mean jersey color from middle 40% of bbox
- Predict cluster + compute centroid distance confidence
- If confidence ≥ 0.85 AND separation adequate → done

### Stage 3: SigLIP (~15ms/crop)
- Extract 768-dim embedding from player crop
- Cosine similarity against tipoff team profiles
- If confidence ≥ 0.80 → done
- Batched: all uncertain players processed in one forward pass

### Stage 4: Qwen2-VL (~100ms/crop)
- Construct prompt from pre-game jersey descriptions
- Visual reasoning on player crop
- Cross-reference with jersey number lookup if number visible
- If answer parseable → done

### Stage 5: Manual review queue
- All stages below threshold → flag for human review
- Target: < 5% of players (baseline: 15-20%)

### Tracking Integration
- DeepSORT/ByteTrack attaches team ID to track, not detection
- Classification travels with the person across frames
- Re-classify only on: confidence drop, frame re-entry, or every N frames

## 4. Scaling to 1000+ Games/Day

### Compute Requirements
| Component | Per-frame cost | Per-game (30 FPS, 48 min) | 1000 games/day |
|-----------|---------------|---------------------------|-----------------|
| YOLO detection | ~5ms | 7.2 min GPU | 120 GPU-hours |
| K-Means (all players) | < 0.1ms | negligible | negligible |
| SigLIP (20% of players) | ~15ms × 2 crops | 2.9 min GPU | 48 GPU-hours |
| Qwen2-VL (5% of players) | ~100ms × 0.5 crops | 2.4 min GPU | 40 GPU-hours |
| **Total** | | ~12.5 min GPU | **~208 GPU-hours** |

### Optimization Levers
- **TensorRT compilation**: 2-3x speedup for SigLIP → cuts SigLIP GPU-hours by 50-60%
- **Cross-frame batching**: Queue uncertain players across 3-5 frames, process together. One SigLIP call every few frames instead of one per frame
- **Dynamic quantization**: Hard games use int8 SigLIP; easy games keep int4. Qwen2-VL uses FP16 in-game, FP32 at tipoff only
- **Temporal consistency**: Most frames need 2-3 active classifications, not all 10 players. Locked players hold classification

### Horizontal Scaling
- GPU pool with load balancer routing games to available GPUs
- Each H100 handles ~5 concurrent games with the cascade
- 1000 games/day ≈ 42 concurrent games at peak → 9 H100s

## 5. Monitoring

### Accuracy Drift
- Track cascade stage distribution over time: if SigLIP escalation rate suddenly spikes, model may need recalibration
- Compare predicted team IDs against manual spot-checks (sampled 1% of games)
- Alert if manual review rate exceeds 10% for any game

### Latency
- P95 end-to-end frame latency — must stay under 100ms
- Per-stage latency breakdown logged per game
- Alert if Qwen2-VL invocation rate exceeds 15% (indicates model degradation or unusually hard game)

### Operational
- GPU utilization and memory pressure per node
- Queue depth for manual review — sustained growth indicates systematic failure
- Description generation API cost tracking (GPT-4o)

## 6. Cost Analysis

### Per-Game Compute Cost
| Item | Cost |
|------|------|
| GPU compute (~12.5 min on H100 at $2/hr) | $0.42 |
| Pre-game description generation (GPT-4o) | $0.04 |
| **Total per game** | **~$0.46** |

### At Scale (1000 games/day)
| Item | Daily | Monthly |
|------|-------|---------|
| GPU compute | $420 | $12,600 |
| GPT-4o descriptions | $40 | $1,200 |
| H100 reserved instances (9 GPUs) | $432 | $12,960 |
| **Total** | **~$460** | **~$14,000** |

Reserved instance pricing is roughly equivalent to pay-per-use at this scale. Reserved instances preferred for predictable load.

## 7. Iteration Plan

### Phase 1: Current Prototype (this submission)
- K-Means → SigLIP → Qwen2-VL cascade
- Improved K-Means via randomized hyperparameter search: 95.3% calibrated average (+16.7% over baseline)
- Pre-game jersey-aware skin filter calibration using reference images
- Manual tipoff calibration
- Evaluated on 3 clips: SigLIP achieves 85.2% average, improved K-Means 95.3%

### Phase 2: Production Hardening (months 1-3)
- Fine-tune SigLIP on accumulated production data (10K+ labeled crops)
- Add jersey number OCR as an independent signal in composite confidence
- Implement DeepSORT temporal consistency
- TensorRT compilation for SigLIP
- Automated tipoff detection (no manual frame selection)
- **Referee prototype library**: Build separate SigLIP referee prototypes for each competition level — NBA (grey pinstripe), NCAA men's (black-white stripe with black raglan sleeves), NCAA women's (predominantly grey), and high school (traditional black-white stripes, varies by state association). The game metadata already specifies competition level; the pre-game pipeline loads the correct referee template automatically. Each prototype is built once from ~20 reference images per level and stored as a static constant — zero ongoing cost
- **Continuous skin filter calibration**: Replace the binary enable/disable with a proportional response — the overlap ratio from the jersey scan becomes a continuous weight on filter aggressiveness (0% overlap → full filtering, 15% → narrowed hue range, 40%+ → disabled). Requires production data across many games to learn the optimal overlap-to-strength mapping; the current binary system is the safe starting point

### Phase 3: Model Distillation (months 3-6)
- Distill Qwen2-VL knowledge into a faster specialist model
- Train a lightweight classification head on SigLIP embeddings (skip cosine similarity)
- Target: eliminate Qwen2-VL stage for 99%+ of games, keeping it only for edge cases
- Expected outcome: < 20ms average per-player latency at > 95% accuracy

### Phase 4: Scale and Generalize (months 6+)
- Extend to other sports (soccer, hockey — different jersey patterns, field layouts)
- Multi-camera support (broadcast vs sideline angles)
- Real-time confidence dashboard for production operators
- A/B testing framework for model upgrades
