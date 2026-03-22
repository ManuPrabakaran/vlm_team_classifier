1. Problem Analysis
The current K-Means system fails 15-20% of the time — not bad luck, but four root causes.
Problem 1: No concept of game difficulty. Same approach applied to every game regardless of jersey color similarity, special editions, lighting, or camera quality.
Problem 2: K-Means is blind to its own uncertainty. Always produces a confident answer regardless of cluster similarity — navy vs black classified at 95% confidence, silently wrong. Centroid separation distance is a measurable proxy: when centroids are too close, every classification of that frame is suspect.
Problem 3: Color is an unstable and incomplete feature. The same jersey looks different under every lighting condition — raw RGB shifts dramatically with exposure, shadows, and compression. In addition, number font, logo placement, stripe pattern, jersey cut, sock design, and court position are all ignored by the current system.
Root Cause Summary: Single unstable signal, no uncertainty awareness, no game-level adaptation. 
2. VLM Model Comparison
VLM embeddings replace mean RGB with a rich 512-1024 dimensional vector capturing everything visually meaningful — same-team players cluster together, different-team players don't. I evaluated four:
Model
Speed (H100)
Memory
Accuracy
Confidence Calibration
Recommended Use
CLIP
~12ms
600MB
Good
Poor
Baseline only — too weak on fine-grained details like logos and number fonts
Florence-2
~50ms
1.5GB
N/A (excluded)
Poor
Excluded — fixed task-token architecture has no contrastive embeddings, no cascade confidence signal, and no three-class output. Architectural mismatch with team classification.
SigLIP
~15ms
900MB
Very Good
Excellent
Primary classifier — best speed/accuracy/calibration balance for constant use
Qwen2-VL
~100ms
8GB
Excellent
Very Good
Escalation classifier — reads jersey numbers, answers targeted visual questions, invoked only when SigLIP is uncertain

Recommendation
SigLIP handles 80%+ of frames at ~15ms. Qwen2-VL handles the hard cases via OCR and targeted visual questioning.
3. Feature Extraction
Rather than a mean RGB value, we extract a full VLM embedding per player crop — capturing color, texture, logo placement, number font, jersey cut, stripe patterns, and sock design simultaneously. Many of these are lighting invariant, directly addressing K-Means' core failure mode.
Jersey Shape from Multiple Angles: Jersey cuts, neckline styles, and pattern layouts are team-specific regardless of lighting or angle. Side and back views still expose shoulder patterns, waistband design, and sock stripes — the VLM extracts structural features raw color clustering can't access.
Logo Placement as a Team Fingerprint: Logo position — centered chest vs upper left, sleeve vs sock — is consistent within a team and differs across teams. Established once at tipoff, it becomes a fast lookup for every subsequent frame.
Context Window: Single frame — keeps latency predictable; temporal consistency handled at the tracking layer.
Two-Phase Feature Extraction: At tipoff, Qwen2-VL runs detailed profile-building queries on every visible player — color, number, logo, stripes, structure. For every subsequent frame, fast SigLIP embeddings compare against those profiles. Qwen2-VL cost paid once; SigLIP handles the other 99%. 
Few-Shot Learning via Tipoff: The tipoff window is our natural few-shot anchor — free labeled examples that arrive automatically with every game, requiring no human annotation. Unlike general image classification tasks that need curated datasets, basketball games provide a built-in calibration window every single time.
4. Implementation Strategy
Pre-Game: Jersey Number Roster Lookup
We build a number lookup table from publicly available roster data before the game starts. Crucially, we flag unique numbers — numbers that only appear on one team. Any player whose number is read clearly as a unique number is automatically classified with near-perfect confidence.
Pre-Game Risk Score
We compute a game difficulty constant — a measure of how likely the known K-Means failure modes are to occur in this specific game, based on jersey color similarity, video and camera quality, special or alternate jersey flags, and lighting conditions. High constant → cascade thresholds tighten, VLM invoked more aggressively.
Cascade Architecture
Stage 0 — Pre-game setup (zero runtime cost)
Build jersey number lookup table, flag unique numbers. 
Apply game risk score to tune cascade thresholds for the entire game.
Stage 1 — Referee detection (near zero cost)
A key insight: **NBA referee uniforms are identical across every game** — the same grey shirt with black vertical stripes, black pants, every arena, every season. Unlike team jerseys, which change per game (home/away, throwbacks, city editions), referee uniforms are a universal constant. This transforms referee detection from a per-game clustering problem into a one-time known-template matching problem that we solve once and deploy forever.

We exploit this by building a **fixed referee prototype** — a pre-computed SigLIP embedding centroid from reference images of NBA referee uniforms, captured from multiple angles (front, back, side). This prototype is stored as a constant and never changes between games. Before any team classification begins, every YOLO detection is compared against the referee prototype via a single cosine similarity check (a dot product on a 768-dim vector — microseconds, effectively free since the SigLIP embedding is already being extracted for team classification downstream).

Two filters make this extremely efficient:
1. **Court position filter**: Referees work the sidelines and baselines — they are rarely positioned near the center of play. By ranking all detected players by x-coordinate and flagging those near the court edges, we can pre-filter candidates before any embedding computation. This repurposes the court position heuristic: instead of (unreliably) guessing which team a player is on based on spatial position, we use spatial position to identify the 2-3 people who are most likely referees — a much more tractable problem.
2. **Bounded cardinality**: There are at most 3 referees on court (usually 2). Once we've found 2-3 referee matches, we stop checking — no more false positives possible.

Detections flagged as referees receive `team_id = -1` and skip the entire classification cascade. This saves compute (2-3 fewer crops through K-Means/SigLIP/Qwen per frame) and prevents referees from polluting team cluster centroids.

Stage 2 — Pre-screening (near zero cost)
 Before invoking any ML model, run three cheap checks:
Court position heuristic: Rank all detected players by x-coordinate relative to each other. Players on the far left are either defensive for the left team or offensive for the right. Cross-reference with known roster roles to resolve which team. Down-weighted during transitions, free throws, and jump balls — returns a prior probability, not a hard assignment.


Cluster separation check: If K-Means centroids are too close together in RGB space, automatically flag all players for VLM escalation regardless of confidence score. Catches the navy-vs-black case before it reaches the classifier.


Lightweight facial re-identification: Not identity matching — just basic appearance features (skin tone, hair, build). Cheap; privacy-preserving continuity signal.


Stage 3 — K-Means with calibrated confidence (fast): Run K-Means on remaining team players (referees already removed), derive confidence from centroid distance ratio. If confidence high AND separation wide AND court position agrees, done.
Stage 4 — SigLIP embedding classification (moderate): Extract embeddings, compare against tipoff team profiles. Lock in if above threshold.
Stage 5 — Qwen2-VL reasoning (expensive, rare): Ask targeted questions — jersey number, logo placement, stripe color. Cross-reference against roster. Build composite confidence.
Stage 6 — Manual review queue (last resort): All stages below threshold → flag for human review. Target: reduce from 15-20% to under 5%.
Multi-Signal Composite Confidence Score
Six independent signals weighted into a single composite score: K-Means color confidence, SigLIP embedding distance from team centroid, court position prior, jersey number lookup match, logo placement consistency, and lightweight facial re-ID. Weights calibrated empirically and adjusted by the game difficulty constant — hard games down-weight color, up-weight number and position. Each signal failing independently doesn't sink the whole classification.
Why We Ruled Things Out / Why We Scaled Things Down
Full facial recognition: Too expensive, unreliable at game resolution, privacy concerns. Ruled out as primary classifier. A lightweight version is included as w6 in the composite score — not identity matching, just "is this the same person I saw two frames ago" using basic appearance features.
Fine-tuning: No labeled dataset yet. Few-shot gets us most of the way there; fine-tuning is the right next step once production data accumulates.
Optimization
Batching and cross-frame batching: All 10 player crops processed simultaneously per frame (5-8x speedup vs sequential). With temporal consistency reducing uncertain players to 2-3 per frame, we queue them across 3-5 frames and batch process together — one SigLIP call every few frames instead of one per frame. No accuracy penalty; locked-in players hold classification while queue fills.
Dynamic quantization: Four context-aware levers — game difficulty (hard games need aggressive int8/int4 to offset increased VLM usage), cascade stage (SigLIP gets int8, Qwen2-VL stays FP16 except tipoff where it runs FP32 for maximum profile accuracy), player confidence (locked-in players can run int4, uncertain players get int8), and time of game (higher precision in first two minutes while profiles stabilize). System self-regulates compute budget rather than applying a blunt global setting.
Temporal consistency via object tracking (DeepSORT/ByteTrack): Attach team ID to the track, not the detection — classification travels with the person across frames. Lock in high-confidence players, only re-classify on confidence drop, frame re-entry, or after N frames. When briefly obscured, hold last known classification and run an embedding distance check on re-emergence. Most frames require active classification of only 2-3 uncertain players, not all 10.
TensorRT/vLLM: For production deployment, compile SigLIP with TensorRT for additional 2-3x inference speedup on H100 hardware.
5. Architecture Diagram

Pre-game: roster lookup + unique number flagging + game difficulty constant + referee prototype (fixed, never changes). Runtime: YOLO bounding boxes → referee filter (SigLIP fixed prototype + sideline position, team_id=-1) → pre-screening (court position, cluster separation, facial re-ID, number lookup) → K-Means → SigLIP → Qwen2-VL → manual review queue → multi-signal confidence fusion → DeepSORT/ByteTrack identification continuity → output.
6. Fallback Strategy
The cascade thresholds are tuned by the pre-game risk score. To be explicit about escalation triggers:
Stay with K-Means when: centroids are well-separated, confidence above 0.85, court position agrees, player has N+ frames of consistent classification.
Escalate to SigLIP when: K-Means confidence below 0.85, centroid separation below threshold, player re-enters frame, or unusual color distribution detected relative to established team profile.
Escalate to Qwen2-VL when: SigLIP confidence below 0.80, two players have similar embeddings with conflicting K-Means assignments, jersey number is potentially readable, or tipoff calibration window.
Escalate to manual review when: all automated stages below threshold, conflicting signals with no clear winner, or novel jersey type with no roster data.
Edge case handling:
Referees — detected before any team classification via fixed SigLIP referee prototype (pre-computed from reference images of the standardized NBA referee uniform). Court position filter (sideline/baseline tendency) pre-screens candidates, and bounded cardinality (max 3 referees) prevents false positives. Referee uniforms are identical across all NBA games — this is a known-template problem solved once, not a per-game clustering problem. Assigned team ID -1 and excluded from the entire classification cascade
Similar jerseys — cluster separation check forces automatic VLM escalation, number lookup becomes tiebreaker
Special edition jerseys — VLM visual reasoning handles naturally, tipoff calibration captures them in team profile on first appearance
7. Risk Assessment
VLM latency at 30 FPS (Medium/High): Cross-frame batching + dynamic quantization + temporal consistency reduces active classifications to 2-3 uncertain players per frame — not all 10. 
Game difficulty constant unavailable (Low/Medium): Fall back to default thresholds. 
Court position heuristic unreliable during transitions, free throws, jump balls (High/Low): Never sole signal, always one weighted input — down-weighted automatically in these situations. 
Jersey numbers not visible or roster data unavailable (Medium/Medium): Fall back gracefully to other signals; tipoff calibration as backup anchor. 
Lighting so bad all signals fail (Low/High): Manual review queue catches this; alert on queue volume spikes. 
GPU memory overflow or model too slow (Medium/High): Load Qwen2-VL on demand, use smaller 2B variant if needed, TensorRT for SigLIP. 
Tipoff window too short or obscured (Low/Medium): Extend window, sample multiple early frames, fall back to zero-shot. 
Model performs poorly on amateur or high school footage (Medium/High): Different training distribution — test explicitly, may require fine-tuning on diverse data.
8. Success Metrics
Classification Accuracy: Frame-by-frame vs ground truth labels, stratified by failure mode. Baseline ~80-85% (K-Means), target >95%.
Latency: Wall clock time from frame input to output, broken down by stage. Target <100ms end-to-end. P95 matters more than average — tail latency kills real-time performance.
Cost per Frame: H100 GPU time per 1000 frames extrapolated to 1000 games/day. Target: 70%+ of frames resolved by K-Means + pre-screening alone.
Manual Review Rate: % of frames escalated to human review. Baseline 15-20%, target <5%.
Confidence Calibration: Reliability diagram on test set — a score of 0.9 should mean ~90% accuracy. Well-calibrated confidence is what makes the cascade trustworthy.
Batch Efficiency: Average batch size per SigLIP call, target 15+ crops via cross-frame batching. Improves naturally as temporal consistency improves.
Ablation Studies: Measure each component incrementally — K-Means only → +cluster separation → +court position → +number lookup → +SigLIP → +Qwen2-VL → +identification continuity → full system. Tells us where to invest future engineering effort.
