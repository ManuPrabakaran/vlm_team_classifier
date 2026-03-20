import numpy as np
from src.baseline import TeamClustering
from src.config import (
    KMEANS_CONFIDENCE_THRESHOLD,
    SIGLIP_CONFIDENCE_THRESHOLD,
    QWEN_CONFIDENCE_THRESHOLD,
    CLUSTER_SEPARATION_MIN,
    GAME_DIFFICULTY_HIGH_THRESHOLD,
    GAME_DIFFICULTY_LOW_THRESHOLD,
    COMPOSITE_WEIGHTS,
    ABLATION_FLAGS,
    N_TEAMS,
    REFEREE_TEAM_ID,
    OUTPUT_METHOD_KMEANS,
    OUTPUT_METHOD_VLM,
    OUTPUT_METHOD_MANUAL,
    TEMPORAL_POSITION_TOLERANCE,
    TEMPORAL_MIN_CONFIDENCE,
    REIDENTIFICATION_INTERVAL,
)

# Lazy import: src.utils pulls in torch/cv2 at module level.
# Functions are imported on first use to keep test collection fast.
_utils = None

def _get_utils():
    global _utils
    if _utils is None:
        import src.utils as _u
        _utils = _u
    return _utils


def _compute_cluster_separation(kmeans_model):
    """Euclidean distance between K-Means centroids in RGB space."""
    c0, c1 = kmeans_model.cluster_centers_
    return float(np.linalg.norm(c0 - c1))


def _compute_centroid_distance_confidence(color, kmeans_model):
    """Confidence from centroid distance ratio. Returns float in (0.5, 1.0]."""
    centers = kmeans_model.cluster_centers_
    d0 = np.linalg.norm(color - centers[0])
    d1 = np.linalg.norm(color - centers[1])
    total = d0 + d1
    if total == 0:
        return 0.5
    predicted = 0 if d0 < d1 else 1
    confidence = (d1 if predicted == 0 else d0) / total
    return float(confidence)


def _bbox_center(bbox):
    """Return (cx, cy) center of a bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


class VLMTeamClassifier:
    """
    Multi-stage cascade classifier: K-Means → SigLIP → Qwen2-VL.

    Augmented with real signals: court position heuristic, jersey number
    roster lookup, CLIP ensemble, and temporal consistency across frames.
    Each signal can be toggled via ABLATION_FLAGS in config.py.
    """

    def __init__(self, config=None):
        config = config or {}
        self.kmeans_threshold = config.get("kmeans_threshold", KMEANS_CONFIDENCE_THRESHOLD)
        self.siglip_threshold = config.get("siglip_threshold", SIGLIP_CONFIDENCE_THRESHOLD)
        self.qwen_threshold = config.get("qwen_threshold", QWEN_CONFIDENCE_THRESHOLD)
        self.separation_min = config.get("separation_min", CLUSTER_SEPARATION_MIN)
        self.composite_weights = config.get("composite_weights", COMPOSITE_WEIGHTS)
        self.ablation = config.get("ablation", ABLATION_FLAGS)

        # Baseline K-Means
        self._kmeans = TeamClustering(n_teams=N_TEAMS)

        # Lazy-loaded VLM models
        self._siglip_model = None
        self._siglip_processor = None
        self._clip_model = None
        self._clip_processor = None
        self._qwen_model = None
        self._qwen_processor = None

        # Team profiles (SigLIP embeddings)
        self._team_profiles = {}  # {0: np.array, 1: np.array}
        # CLIP team profiles for ensemble
        self._clip_team_profiles = {}  # {0: np.array, 1: np.array}

        # Game context
        self._cluster_separation = None
        self._game_difficulty = None
        self._team_names = None
        self._special_jersey = None
        self._jersey_descriptions = None

        # Roster for jersey number lookup: {number_str: team_id}
        # Only contains numbers unique to one team
        self._unique_numbers = {}
        self._roster = {0: set(), 1: set()}

        # Temporal consistency: track cache keyed by approximate bbox center
        # Each entry: {"team_id": int, "confidence": float, "frame_count": int}
        self._track_cache = {}
        self._frame_number = 0

        self._device = None  # Resolved lazily on first model load

        # Cascade stats
        self._cascade_stats = {
            "kmeans": 0, "siglip": 0, "qwen": 0, "manual": 0,
            "temporal": 0, "number_lookup": 0,
        }

    # ------------------------------------------------------------------ #
    # Stage 0: Pre-game setup
    # ------------------------------------------------------------------ #

    def set_game_context(self, team_names=None, special_jersey=None,
                         jersey_descriptions=None):
        self._team_names = team_names
        self._special_jersey = special_jersey or {}
        self._jersey_descriptions = jersey_descriptions

    def set_roster(self, team_0_numbers, team_1_numbers):
        """
        Set jersey number rosters for both teams.
        Automatically identifies unique numbers (only on one team).

        Args:
            team_0_numbers: set/list of jersey number strings for team 0
            team_1_numbers: set/list of jersey number strings for team 1
        """
        self._roster[0] = set(str(n) for n in team_0_numbers)
        self._roster[1] = set(str(n) for n in team_1_numbers)
        # Find numbers unique to one team — these are high-confidence signals
        self._unique_numbers = {}
        for n in self._roster[0]:
            if n not in self._roster[1]:
                self._unique_numbers[n] = 0
        for n in self._roster[1]:
            if n not in self._roster[0]:
                self._unique_numbers[n] = 1

    def _compute_game_difficulty(self):
        """
        Compute game difficulty from cluster separation and metadata.
        Returns float in [0, 1]. Higher = harder game.
        """
        if self._cluster_separation is None:
            self._game_difficulty = 0.5
            return self._game_difficulty

        max_separation = 200.0
        raw = 1.0 - min(self._cluster_separation / max_separation, 1.0)

        if self._special_jersey and any(self._special_jersey.values()):
            raw = min(raw + 0.1, 1.0)

        self._game_difficulty = round(raw, 3)
        return self._game_difficulty

    # ------------------------------------------------------------------ #
    # Stage 1: Pre-screening
    # ------------------------------------------------------------------ #

    def _check_cluster_separation(self):
        """Returns True if K-Means centroids are far enough apart."""
        if self._cluster_separation is None:
            return False
        return self._cluster_separation >= self.separation_min

    def _court_position_prior(self, bbox, all_bboxes):
        """
        Cheap heuristic: group players by x-position into two halves.

        Currently weighted at 0.0 in COMPOSITE_WEIGHTS — the infrastructure
        is built but the signal carries no influence until we can ground it
        in real positional understanding.

        The naive version (left-half vs right-half of detected players) is
        unreliable because basketball court positioning is far more complex
        than a simple spatial split. To make this signal meaningful, we'd
        need several layers of context:

        1. **Court geometry awareness**: Homography mapping from the camera
           frame to a standardized court coordinate system. Without this,
           "left side of the frame" means nothing — camera angle, zoom, and
           broadcast crop all shift where the midcourt line appears.

        2. **Play state detection**: Court position is only informative
           during settled half-court play (5v5 offense/defense). During
           transitions, fast breaks, inbound plays, and free throws,
           players from both teams intermix spatially. A play-state
           classifier would gate when this signal is active.

        3. **Role-aware positional priors**: A point guard, a center, and
           a wing occupy systematically different court regions. If we knew
           each player's role (via roster metadata or pose estimation),
           we could build per-role spatial distributions. A center near
           the paint is expected; a center at the three-point line is
           unusual and weakens the positional prior.

        4. **Possession tracking**: The offensive team's spatial
           distribution differs from the defensive team's. Knowing which
           team has possession (from game clock data, ball tracking, or
           broadcast graphics) would flip the expected spatial mapping.

        Until these layers exist, this method returns a weak guess at best.
        The weight is zeroed out so it doesn't pollute the composite score,
        but the code path remains for future iterations where court-aware
        features become available.

        Returns (team_id, confidence) or (None, 0.0) if too few players.
        """
        if not self.ablation.get("court_position", True):
            return (None, 0.0)

        if all_bboxes is None or len(all_bboxes) < 4:
            return (None, 0.0)

        # Get x-centers of all players
        centers = [_bbox_center(b)[0] for b in all_bboxes]
        this_x = _bbox_center(bbox)[0]

        median_x = float(np.median(centers))

        # If this player is near the median, the signal is weak — skip
        spread = max(centers) - min(centers)
        if spread < 100:  # Players too bunched up (transition play, free throw)
            return (None, 0.0)

        distance_from_median = abs(this_x - median_x)
        if distance_from_median < spread * 0.15:
            return (None, 0.0)

        # Assign team based on which side of the median
        team_id = 0 if this_x < median_x else 1

        # Confidence scales with distance from median, capped at 0.65
        # This is deliberately weak — court position is unreliable during
        # transitions, fast breaks, and set pieces
        confidence = min(0.55 + 0.10 * (distance_from_median / (spread / 2)), 0.65)

        return (team_id, round(confidence, 3))

    def _check_jersey_number(self, number_str):
        """
        Look up a jersey number against the roster.
        Returns (team_id, confidence) if the number is unique to one team,
        or (None, 0.0) if not found or ambiguous.

        The lookup itself is a trivial dict check — the expensive part is
        the upstream OCR that produces the number string. At per-second
        sampling, fewer than 1% of crops show a legible jersey number
        (player facing away, motion blur, occlusion). Running OCR on every
        crop would waste compute for a signal that fires rarely.

        The practical strategy: in a continuous-frame pipeline (e.g. 30fps
        ingestion), opportunistically attempt OCR on a sparse subset of
        frames — perhaps every 30th frame, or only when a crop's sharpness
        score exceeds a threshold. When a number IS clearly visible, that
        single successful read propagates through the temporal cache and
        locks the player's team assignment for dozens of subsequent frames
        without any further OCR. One good read at frame 120 can carry
        through frames 121–150+ at zero marginal cost.

        This makes jersey number lookup high-value despite low per-frame
        hit rate: the cost is amortized across the temporal window, and
        the 0.99 confidence of a unique roster match means the lock is
        rarely overridden by weaker signals.

        Assumes the caller provides number_str from an external OCR system
        (e.g. SmolVLM2, which is already in the Paloa pipeline for jersey
        reading). The classifier does not run OCR itself — it only
        consumes the result.
        """
        if not self.ablation.get("number_lookup", True):
            return (None, 0.0)

        if number_str is None or not self._unique_numbers:
            return (None, 0.0)

        number_str = str(number_str).strip()
        if number_str in self._unique_numbers:
            return (self._unique_numbers[number_str], 0.99)

        return (None, 0.0)

    def _check_temporal_cache(self, bbox):
        """
        Check if we've seen a player at approximately this position before.
        Returns (team_id, confidence) if cached and still valid,
        or (None, 0.0) if no match.
        """
        if not self.ablation.get("temporal_consistency", True):
            return (None, 0.0)

        cx, cy = _bbox_center(bbox)
        tol = TEMPORAL_POSITION_TOLERANCE

        for (cached_cx, cached_cy), entry in self._track_cache.items():
            if abs(cx - cached_cx) < tol and abs(cy - cached_cy) < tol:
                # Check if this cache entry is still fresh
                frames_since = self._frame_number - entry["last_frame"]
                if frames_since > REIDENTIFICATION_INTERVAL:
                    continue
                if entry["confidence"] >= TEMPORAL_MIN_CONFIDENCE:
                    # Slight confidence decay over frames
                    decay = max(0.0, 0.01 * frames_since)
                    conf = max(entry["confidence"] - decay, 0.5)
                    return (entry["team_id"], round(conf, 3))

        return (None, 0.0)

    def _update_temporal_cache(self, bbox, team_id, confidence):
        """Update the temporal cache with a new prediction."""
        if not self.ablation.get("temporal_consistency", True):
            return

        cx, cy = _bbox_center(bbox)
        tol = TEMPORAL_POSITION_TOLERANCE

        # Find existing entry at similar position and update it
        for (cached_cx, cached_cy) in list(self._track_cache.keys()):
            if abs(cx - cached_cx) < tol and abs(cy - cached_cy) < tol:
                del self._track_cache[(cached_cx, cached_cy)]
                break

        self._track_cache[(cx, cy)] = {
            "team_id": team_id,
            "confidence": confidence,
            "last_frame": self._frame_number,
        }

    def advance_frame(self):
        """Call between frames to advance the temporal counter."""
        self._frame_number += 1

    # ------------------------------------------------------------------ #
    # Stages 2–4: Classification cascade
    # ------------------------------------------------------------------ #

    def fit(self, frame, bboxes, team_ids=None):
        """
        Tipoff calibration from initial frame(s).

        1. Fit K-Means on jersey colors
        2. Compute cluster separation → game difficulty
        3. Build SigLIP team embedding profiles if separation is low
        4. Build CLIP team profiles if ensemble enabled
        """
        self._kmeans.fit(frame, bboxes)
        self._cluster_separation = _compute_cluster_separation(self._kmeans.kmeans)
        self._compute_game_difficulty()

        if not self._check_cluster_separation() or team_ids is not None:
            self._ensure_siglip()
            self._build_team_profiles(frame, bboxes, team_ids)
            if self.ablation.get("clip_ensemble", True):
                self._ensure_clip()
                self._build_clip_profiles(frame, bboxes, team_ids)

    def _build_team_profiles(self, frame, bboxes, team_ids=None):
        """Build SigLIP embedding centroids per team from tipoff frame."""
        embeddings = {0: [], 1: []}

        for i, bbox in enumerate(bboxes):
            crop = _get_utils().crop_player(frame, bbox)
            if crop.size == 0:
                continue
            emb = _get_utils().extract_siglip_embedding(
                crop, self._siglip_model, self._siglip_processor, self._device
            )

            if team_ids is not None:
                tid = team_ids[i]
                if tid in embeddings:
                    embeddings[tid].append(emb)
            else:
                tid = self._kmeans.predict_team(frame, bbox)
                embeddings[tid].append(emb)

        for tid in embeddings:
            if embeddings[tid]:
                centroid = np.mean(embeddings[tid], axis=0)
                self._team_profiles[tid] = centroid / np.linalg.norm(centroid)

    def _build_clip_profiles(self, frame, bboxes, team_ids=None):
        """Build CLIP embedding centroids per team from tipoff frame."""
        embeddings = {0: [], 1: []}

        for i, bbox in enumerate(bboxes):
            crop = _get_utils().crop_player(frame, bbox)
            if crop.size == 0:
                continue
            emb = _get_utils().extract_clip_embedding(
                crop, self._clip_model, self._clip_processor, self._device
            )

            if team_ids is not None:
                tid = team_ids[i]
                if tid in embeddings:
                    embeddings[tid].append(emb)
            else:
                tid = self._kmeans.predict_team(frame, bbox)
                embeddings[tid].append(emb)

        for tid in embeddings:
            if embeddings[tid]:
                centroid = np.mean(embeddings[tid], axis=0)
                self._clip_team_profiles[tid] = centroid / np.linalg.norm(centroid)

    def predict(self, frame, bbox, all_bboxes=None, jersey_number=None) -> dict:
        """
        Classify a single player through the full cascade.

        Returns:
            {
                "team_id": int,       # 0, 1, or -1 (referee)
                "confidence": float,  # composite confidence
                "method": str,        # which stage resolved it
                "signals": dict       # individual signal values
            }
        """
        signals = {}

        # ── Check temporal cache first (cheapest possible) ──
        temp_team, temp_conf = self._check_temporal_cache(bbox)
        if temp_team is not None and temp_conf >= TEMPORAL_MIN_CONFIDENCE:
            signals["temporal"] = {"team_id": temp_team, "confidence": temp_conf}
            self._cascade_stats["temporal"] += 1
            self._update_temporal_cache(bbox, temp_team, temp_conf)
            return self._build_result(temp_team, temp_conf, "temporal", signals)

        # ── Check jersey number if provided (near-perfect signal) ──
        if jersey_number is not None:
            num_team, num_conf = self._check_jersey_number(jersey_number)
            if num_team is not None:
                signals["number_lookup"] = {"team_id": num_team, "confidence": num_conf}
                self._cascade_stats["number_lookup"] += 1
                self._update_temporal_cache(bbox, num_team, num_conf)
                return self._build_result(num_team, num_conf, "number_lookup", signals)

        # ── Stage 1: Pre-screening ──
        if all_bboxes is not None:
            pos_team, pos_conf = self._court_position_prior(bbox, all_bboxes)
            if pos_team is not None:
                signals["court_position"] = {"team_id": pos_team, "confidence": pos_conf}

        separation_ok = self._check_cluster_separation()

        # ── Stage 2: K-Means ──
        km_team, km_conf = self._predict_kmeans(frame, bbox)
        signals["kmeans"] = {"team_id": km_team, "confidence": km_conf}

        # K-Means only short-circuits when BOTH separation is very high AND
        # confidence is near-certain. This prevents K-Means from stealing
        # predictions from SigLIP when it's confidently wrong (~87% accuracy
        # vs SigLIP's 97.8%). Most predictions should flow to SigLIP.
        if separation_ok and km_conf >= 0.95 and self._cluster_separation >= self.separation_min * 2:
            self._cascade_stats["kmeans"] += 1
            composite = self._compute_composite_confidence(signals)
            self._update_temporal_cache(bbox, km_team, composite)
            return self._build_result(km_team, composite, OUTPUT_METHOD_KMEANS, signals)

        # ── Stage 3: SigLIP (primary classifier) ──
        # SigLIP always resolves when team profiles exist — it's the
        # strongest model. No confidence gating: the individual eval shows
        # 97.8% on clip1 by simply picking the closest centroid. Gating on
        # the cosine-distance confidence ratio was blocking correct
        # predictions (confidence ~0.55-0.70 vs threshold 0.80).
        if self._team_profiles:
            self._ensure_siglip()
            sig_team, sig_conf = self._predict_siglip(frame, bbox)
            signals["siglip"] = {"team_id": sig_team, "confidence": sig_conf}

            # ── CLIP fallback: only when SigLIP confidence is low ──
            # CLIP beats SigLIP on clip3_edge (78.1% vs 71.9%), so we
            # check CLIP when SigLIP is uncertain and let it override.
            if sig_conf < self.siglip_threshold:
                if self._clip_team_profiles and self.ablation.get("clip_ensemble", True):
                    clip_team, clip_conf = self._predict_clip(frame, bbox)
                    signals["clip"] = {"team_id": clip_team, "confidence": clip_conf}

                    if clip_conf > sig_conf:
                        sig_team = clip_team
                        sig_conf = clip_conf
                    elif sig_team == clip_team:
                        sig_conf = min(sig_conf + 0.05, 1.0)

                    self._cascade_stats["siglip"] += 1
                    composite = self._compute_composite_confidence(signals)
                    self._update_temporal_cache(bbox, sig_team, composite)
                    return self._build_result(sig_team, composite, OUTPUT_METHOD_VLM + ":clip_fallback", signals)

            # SigLIP resolves — no threshold gate
            self._cascade_stats["siglip"] += 1
            composite = self._compute_composite_confidence(signals)
            self._update_temporal_cache(bbox, sig_team, composite)
            return self._build_result(sig_team, composite, OUTPUT_METHOD_VLM + ":siglip", signals)

        # ── Stage 4: Qwen2-VL (only if no SigLIP profiles) ──
        if self._jersey_descriptions:
            self._ensure_qwen()
            qw_team, qw_conf = self._predict_qwen(frame, bbox)
            signals["qwen"] = {"team_id": qw_team, "confidence": qw_conf}

            if qw_team is not None:
                self._cascade_stats["qwen"] += 1
                composite = self._compute_composite_confidence(signals)
                self._update_temporal_cache(bbox, qw_team, composite)
                return self._build_result(qw_team, composite, OUTPUT_METHOD_VLM + ":qwen", signals)

        # ── Stage 5: Manual review fallback ──
        self._cascade_stats["manual"] += 1
        best_team, best_conf = self._best_available_signal(signals)
        self._update_temporal_cache(bbox, best_team, best_conf)
        return self._build_result(best_team, best_conf, OUTPUT_METHOD_MANUAL, signals)

    def predict_batch(self, frame, bboxes, jersey_numbers=None) -> list:
        """Batch prediction — all players in one frame."""
        results = []
        for i, bbox in enumerate(bboxes):
            number = jersey_numbers[i] if jersey_numbers else None
            results.append(self.predict(frame, bbox, all_bboxes=bboxes,
                                        jersey_number=number))
        return results

    # ------------------------------------------------------------------ #
    # Individual stage predictors
    # ------------------------------------------------------------------ #

    def _predict_kmeans(self, frame, bbox):
        """Stage 2: K-Means + calibrated confidence."""
        color = self._kmeans.extract_jersey_color(frame, bbox)
        team_id = int(self._kmeans.kmeans.predict([color])[0])
        confidence = _compute_centroid_distance_confidence(color, self._kmeans.kmeans)
        return team_id, confidence

    def _predict_siglip(self, frame, bbox):
        """Stage 3: SigLIP embedding distance to team profiles."""
        u = _get_utils()
        crop = u.crop_player(frame, bbox)
        if crop.size == 0:
            return 0, 0.5
        emb = u.extract_siglip_embedding(
            crop, self._siglip_model, self._siglip_processor, self._device
        )
        distances = u.compute_embedding_distance(emb, self._team_profiles)
        team_id = min(distances, key=distances.get)
        total = sum(distances.values())
        if total == 0:
            return team_id, 0.5
        confidence = 1.0 - (distances[team_id] / total)
        return team_id, float(confidence)

    def _predict_clip(self, frame, bbox):
        """Stage 3b: CLIP embedding distance to team profiles (ensemble)."""
        u = _get_utils()
        crop = u.crop_player(frame, bbox)
        if crop.size == 0:
            return 0, 0.5
        emb = u.extract_clip_embedding(
            crop, self._clip_model, self._clip_processor, self._device
        )
        distances = u.compute_embedding_distance(emb, self._clip_team_profiles)
        team_id = min(distances, key=distances.get)
        total = sum(distances.values())
        if total == 0:
            return team_id, 0.5
        confidence = 1.0 - (distances[team_id] / total)
        return team_id, float(confidence)

    def _predict_qwen(self, frame, bbox):
        """Stage 4: Qwen2-VL visual reasoning with jersey descriptions."""
        crop = _get_utils().crop_player(frame, bbox)
        if crop.size == 0:
            return None, 0.0

        from PIL import Image
        import cv2
        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        desc_0 = self._jersey_descriptions.get(0, "Team 0")
        desc_1 = self._jersey_descriptions.get(1, "Team 1")

        prompt = (
            f"Look at this basketball player image. "
            f"Team 0 wears: {desc_0}. "
            f"Team 1 wears: {desc_1}. "
            f"Which team does this player belong to? "
            f"Output only 0 or 1."
        )

        from qwen_vl_utils import process_vision_info
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._qwen_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        import torch
        with torch.no_grad():
            generated_ids = self._qwen_model.generate(**inputs, max_new_tokens=10)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self._qwen_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )[0].strip()

        if "0" in output and "1" not in output:
            return 0, self.qwen_threshold
        elif "1" in output and "0" not in output:
            return 1, self.qwen_threshold
        elif output.startswith("0"):
            return 0, self.qwen_threshold
        elif output.startswith("1"):
            return 1, self.qwen_threshold
        else:
            return None, 0.0

    # ------------------------------------------------------------------ #
    # Lazy model loading
    # ------------------------------------------------------------------ #

    def _ensure_siglip(self):
        if self._siglip_model is None:
            if self._device is None:
                self._device = _get_utils().get_device()
            self._siglip_model, self._siglip_processor = _get_utils().load_siglip_model(self._device)

    def _ensure_clip(self):
        if self._clip_model is None:
            if self._device is None:
                self._device = _get_utils().get_device()
            self._clip_model, self._clip_processor = _get_utils().load_clip_model(self._device)

    def _ensure_qwen(self):
        if self._qwen_model is None:
            if self._device is None:
                self._device = _get_utils().get_device()
            self._qwen_model, self._qwen_processor = _get_utils().load_qwen_model(self._device)

    # ------------------------------------------------------------------ #
    # Composite confidence
    # ------------------------------------------------------------------ #

    def _compute_composite_confidence(self, signals):
        """
        Weighted combination of available signals.
        Only includes signals that were actually computed.
        Weights renormalized to sum to 1.0 over active signals.
        """
        active_weight = 0.0
        weighted_conf = 0.0

        for key, weight in self.composite_weights.items():
            if key in signals:
                active_weight += weight
                weighted_conf += weight * signals[key]["confidence"]

        if active_weight == 0:
            return 0.5
        return weighted_conf / active_weight

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _build_result(self, team_id, confidence, method, signals):
        return {
            "team_id": int(team_id) if team_id is not None else REFEREE_TEAM_ID,
            "confidence": round(float(confidence), 4),
            "method": method,
            "signals": signals,
        }

    def _best_available_signal(self, signals):
        """Pick the highest-confidence signal as fallback."""
        best_team, best_conf = 0, 0.0
        for key in ["qwen", "clip", "siglip", "kmeans", "court_position"]:
            if key in signals and signals[key]["confidence"] > best_conf:
                best_team = signals[key]["team_id"]
                best_conf = signals[key]["confidence"]
        return best_team, best_conf

    # ------------------------------------------------------------------ #
    # Inspection
    # ------------------------------------------------------------------ #

    def get_team_profiles(self):
        return dict(self._team_profiles)

    def get_game_difficulty(self):
        return self._game_difficulty

    def get_cascade_stats(self):
        return dict(self._cascade_stats)

    def get_cluster_separation(self):
        return self._cluster_separation

    def get_roster(self):
        return dict(self._roster)

    def get_unique_numbers(self):
        return dict(self._unique_numbers)

    def get_track_cache_size(self):
        return len(self._track_cache)
