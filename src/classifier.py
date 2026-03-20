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
    N_TEAMS,
    REFEREE_TEAM_ID,
    OUTPUT_METHOD_KMEANS,
    OUTPUT_METHOD_VLM,
    OUTPUT_METHOD_MANUAL,
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


class VLMTeamClassifier:
    """
    Multi-stage cascade classifier: K-Means → SigLIP → Qwen2-VL.

    Each stage is progressively more expensive but more accurate.
    The cascade short-circuits as soon as confidence exceeds the
    threshold for the current stage, keeping average latency low.
    """

    def __init__(self, config=None):
        config = config or {}
        self.kmeans_threshold = config.get("kmeans_threshold", KMEANS_CONFIDENCE_THRESHOLD)
        self.siglip_threshold = config.get("siglip_threshold", SIGLIP_CONFIDENCE_THRESHOLD)
        self.qwen_threshold = config.get("qwen_threshold", QWEN_CONFIDENCE_THRESHOLD)
        self.separation_min = config.get("separation_min", CLUSTER_SEPARATION_MIN)
        self.composite_weights = config.get("composite_weights", COMPOSITE_WEIGHTS)

        # Baseline K-Means
        self._kmeans = TeamClustering(n_teams=N_TEAMS)

        # Lazy-loaded VLM models
        self._siglip_model = None
        self._siglip_processor = None
        self._qwen_model = None
        self._qwen_processor = None

        # Team profiles (SigLIP embeddings)
        self._team_profiles = {}  # {0: np.array, 1: np.array}

        # Game context
        self._cluster_separation = None
        self._game_difficulty = None
        self._team_names = None
        self._special_jersey = None
        self._jersey_descriptions = None

        self._device = None  # Resolved lazily on first model load

        # Cascade stats
        self._cascade_stats = {"kmeans": 0, "siglip": 0, "qwen": 0, "manual": 0}

    # ------------------------------------------------------------------ #
    # Stage 0: Pre-game setup
    # ------------------------------------------------------------------ #

    def set_game_context(self, team_names=None, special_jersey=None,
                         jersey_descriptions=None):
        """
        Pre-game configuration.

        Args:
            team_names: dict {0: "Team A", 1: "Team B"} or None
            special_jersey: dict {0: False, 1: True} flags for alternate kits
            jersey_descriptions: dict {0: str, 1: str} for Qwen2-VL prompting
        """
        self._team_names = team_names
        self._special_jersey = special_jersey or {}
        self._jersey_descriptions = jersey_descriptions

    def _compute_game_difficulty(self):
        """
        Compute game difficulty from cluster separation and metadata.

        Returns float in [0, 1]. Higher = harder game.
        - Low separation → high difficulty
        - Special jerseys → bump difficulty
        """
        if self._cluster_separation is None:
            self._game_difficulty = 0.5
            return self._game_difficulty

        # Normalize separation: 0 (identical) to ~200 (max RGB distance)
        # Map to difficulty: high separation → low difficulty
        max_separation = 200.0
        raw = 1.0 - min(self._cluster_separation / max_separation, 1.0)

        # Bump for special jerseys
        if self._special_jersey and any(self._special_jersey.values()):
            raw = min(raw + 0.1, 1.0)

        self._game_difficulty = round(raw, 3)
        return self._game_difficulty

    # ------------------------------------------------------------------ #
    # Stage 1: Pre-screening
    # ------------------------------------------------------------------ #

    def _check_cluster_separation(self):
        """
        Check if K-Means centroids are far enough apart.
        Returns True if separation is adequate, False if VLM escalation needed.
        """
        if self._cluster_separation is None:
            return False
        return self._cluster_separation >= self.separation_min

    def _court_position_prior(self, bbox, all_bboxes):
        """
        Cheap heuristic: relative x-position among all players.

        Stub — returns (None, 0). Documented as future production work.
        Court position is unreliable during transitions, free throws, and
        jump balls, so it's down-weighted in those scenarios.
        """
        return (None, 0.0)

    # ------------------------------------------------------------------ #
    # Stages 2–4: Classification cascade
    # ------------------------------------------------------------------ #

    def fit(self, frame, bboxes, team_ids=None):
        """
        Tipoff calibration from initial frame(s).

        1. Fit K-Means on jersey colors
        2. Compute cluster separation → game difficulty
        3. Build SigLIP team embedding profiles if separation is low

        Args:
            frame: BGR numpy array
            bboxes: list of [x1, y1, x2, y2]
            team_ids: optional list of known team IDs for supervised fitting
        """
        # Stage 2 init: K-Means
        self._kmeans.fit(frame, bboxes)
        self._cluster_separation = _compute_cluster_separation(self._kmeans.kmeans)
        self._compute_game_difficulty()

        # Stage 3 init: SigLIP profiles if separation is low or team_ids given
        if not self._check_cluster_separation() or team_ids is not None:
            self._ensure_siglip()
            self._build_team_profiles(frame, bboxes, team_ids)

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
                # Use K-Means assignment
                tid = self._kmeans.predict_team(frame, bbox)
                embeddings[tid].append(emb)

        for tid in embeddings:
            if embeddings[tid]:
                centroid = np.mean(embeddings[tid], axis=0)
                self._team_profiles[tid] = centroid / np.linalg.norm(centroid)

    def predict(self, frame, bbox, all_bboxes=None) -> dict:
        """
        Classify a single player through the cascade.

        Returns:
            {
                "team_id": int,       # 0, 1, or -1 (referee)
                "confidence": float,  # composite confidence
                "method": str,        # which stage resolved it
                "signals": dict       # individual signal values
            }
        """
        signals = {}

        # Stage 1: Pre-screening
        if all_bboxes is not None:
            pos_team, pos_conf = self._court_position_prior(bbox, all_bboxes)
            if pos_team is not None:
                signals["court_position"] = {"team_id": pos_team, "confidence": pos_conf}

        separation_ok = self._check_cluster_separation()

        # Stage 2: K-Means
        km_team, km_conf = self._predict_kmeans(frame, bbox)
        signals["kmeans"] = {"team_id": km_team, "confidence": km_conf}

        if separation_ok and km_conf >= self.kmeans_threshold:
            self._cascade_stats["kmeans"] += 1
            return self._build_result(km_team, km_conf, OUTPUT_METHOD_KMEANS, signals)

        # Stage 3: SigLIP
        if self._team_profiles:
            self._ensure_siglip()
            sig_team, sig_conf = self._predict_siglip(frame, bbox)
            signals["siglip"] = {"team_id": sig_team, "confidence": sig_conf}

            if sig_conf >= self.siglip_threshold:
                self._cascade_stats["siglip"] += 1
                composite = self._compute_composite_confidence(signals)
                return self._build_result(sig_team, composite, OUTPUT_METHOD_VLM + ":siglip", signals)

        # Stage 4: Qwen2-VL
        if self._jersey_descriptions:
            self._ensure_qwen()
            qw_team, qw_conf = self._predict_qwen(frame, bbox)
            signals["qwen"] = {"team_id": qw_team, "confidence": qw_conf}

            if qw_team is not None:
                self._cascade_stats["qwen"] += 1
                composite = self._compute_composite_confidence(signals)
                return self._build_result(qw_team, composite, OUTPUT_METHOD_VLM + ":qwen", signals)

        # Stage 5: Manual review fallback
        self._cascade_stats["manual"] += 1
        best_team, best_conf = self._best_available_signal(signals)
        return self._build_result(best_team, best_conf, OUTPUT_METHOD_MANUAL, signals)

    def predict_batch(self, frame, bboxes) -> list:
        """Batch prediction — all players in one frame."""
        return [self.predict(frame, bbox, all_bboxes=bboxes) for bbox in bboxes]

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
        # Convert distance to confidence: closer = higher confidence
        total = sum(distances.values())
        if total == 0:
            return team_id, 0.5
        confidence = 1.0 - (distances[team_id] / total)
        return team_id, float(confidence)

    def _predict_qwen(self, frame, bbox):
        """
        Stage 4: Qwen2-VL visual reasoning with jersey descriptions.

        Constructs a prompt asking Qwen to classify the player crop
        based on the pre-game jersey descriptions.
        """
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
        # Strip input tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self._qwen_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )[0].strip()

        # Parse output
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
        """Lazy-load SigLIP model on first use."""
        if self._siglip_model is None:
            if self._device is None:
                self._device = _get_utils().get_device()
            self._siglip_model, self._siglip_processor = _get_utils().load_siglip_model(self._device)

    def _ensure_qwen(self):
        """Lazy-load Qwen2-VL on first use."""
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
        for key in ["qwen", "siglip", "kmeans"]:
            if key in signals and signals[key]["confidence"] > best_conf:
                best_team = signals[key]["team_id"]
                best_conf = signals[key]["confidence"]
        return best_team, best_conf

    # ------------------------------------------------------------------ #
    # Inspection
    # ------------------------------------------------------------------ #

    def get_team_profiles(self):
        """Return current SigLIP team embedding profiles."""
        return dict(self._team_profiles)

    def get_game_difficulty(self):
        """Return computed game difficulty constant."""
        return self._game_difficulty

    def get_cascade_stats(self):
        """Return counts of how many predictions resolved at each stage."""
        return dict(self._cascade_stats)

    def get_cluster_separation(self):
        """Return K-Means centroid separation distance."""
        return self._cluster_separation
