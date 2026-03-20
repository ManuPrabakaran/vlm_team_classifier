import json
import numpy as np
import pytest
from pathlib import Path
from src.classifier import VLMTeamClassifier
from src.baseline import TeamClustering


DATA_DIR = Path(__file__).parent.parent / "data"


def load_ground_truth(clip_name):
    gt_path = DATA_DIR / f"{clip_name}_ground_truth.json"
    if not gt_path.exists():
        pytest.skip(f"Ground truth not found: {gt_path}")
    with open(gt_path) as f:
        return json.load(f)


def make_synthetic_frame(ground_truth, width=1920, height=1080):
    """
    Create a synthetic frame with colored rectangles at bbox locations.
    Team 0 = red, Team 1 = blue. Tests that the classifier can separate
    them even on synthetic data.
    """
    frame = np.full((height, width, 3), 128, dtype=np.uint8)  # Gray background

    if not ground_truth:
        return frame, [], []

    first_frame = ground_truth[0]
    bboxes = []
    team_ids = []

    for label in first_frame.get("labels", []):
        if not label.get("valid", True):
            continue
        if label.get("team_id", -1) == -1:
            continue

        bbox = label["bbox"]
        x1, y1, x2, y2 = bbox
        # Clamp to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        tid = label["team_id"]
        color = [0, 0, 255] if tid == 0 else [255, 0, 0]  # BGR: red vs blue
        frame[y1:y2, x1:x2] = color

        bboxes.append([x1, y1, x2, y2])
        team_ids.append(tid)

    return frame, bboxes, team_ids


class TestSyntheticClassification:
    """Test classifier on synthetic colored frames derived from real bbox layouts."""

    @pytest.mark.parametrize("clip_name", ["clip1_easy", "clip2_hard", "clip3_edge"])
    def test_kmeans_on_synthetic(self, clip_name):
        gt = load_ground_truth(clip_name)
        frame, bboxes, team_ids = make_synthetic_frame(gt)

        if len(bboxes) < 2:
            pytest.skip(f"Not enough valid bboxes in {clip_name}")

        tc = TeamClustering()
        tc.fit(frame, bboxes)

        predictions = [tc.predict_team(frame, bbox) for bbox in bboxes]

        # On synthetic red/blue, both label orientations should give 100%
        # Try both orientations
        acc_direct = sum(p == t for p, t in zip(predictions, team_ids)) / len(team_ids)
        acc_flipped = sum(p != t for p, t in zip(predictions, team_ids)) / len(team_ids)
        accuracy = max(acc_direct, acc_flipped)

        assert accuracy >= 0.9, f"{clip_name}: synthetic accuracy {accuracy:.1%} < 90%"

    @pytest.mark.parametrize("clip_name", ["clip1_easy", "clip2_hard", "clip3_edge"])
    def test_cascade_on_synthetic(self, clip_name):
        gt = load_ground_truth(clip_name)
        frame, bboxes, team_ids = make_synthetic_frame(gt)

        if len(bboxes) < 2:
            pytest.skip(f"Not enough valid bboxes in {clip_name}")

        clf = VLMTeamClassifier()
        clf.fit(frame, bboxes)

        results = clf.predict_batch(frame, bboxes)

        # Should resolve at kmeans or manual (no SigLIP profiles in synthetic)
        for r in results:
            assert r["method"] in ("kmeans", "manual")
            assert r["confidence"] > 0.5

        # Check accuracy
        preds = [r["team_id"] for r in results]
        acc_direct = sum(p == t for p, t in zip(preds, team_ids)) / len(team_ids)
        acc_flipped = sum((1 - p) == t for p, t in zip(preds, team_ids)) / len(team_ids)
        accuracy = max(acc_direct, acc_flipped)

        assert accuracy >= 0.9


class TestCascadeStatsTracking:
    def test_stats_accumulate(self):
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        frame[:, :200] = [0, 0, 255]
        frame[:, 200:] = [255, 0, 0]
        bboxes = [[50, 20, 150, 180], [250, 20, 350, 180]]

        clf = VLMTeamClassifier()
        clf.fit(frame, bboxes)
        clf.predict_batch(frame, bboxes)

        stats = clf.get_cascade_stats()
        assert stats["kmeans"] == 2
        assert stats["siglip"] == 0
        assert stats["qwen"] == 0
        assert stats["manual"] == 0


class TestGameDifficultyIntegration:
    def test_easy_game_low_difficulty(self):
        """Red vs blue should produce low game difficulty."""
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        frame[:, :200] = [0, 0, 255]
        frame[:, 200:] = [255, 0, 0]
        bboxes = [[50, 20, 150, 180], [250, 20, 350, 180]]

        clf = VLMTeamClassifier()
        clf.fit(frame, bboxes)

        assert clf.get_game_difficulty() < 0.3
        assert clf.get_cluster_separation() > 100

    def test_hard_game_high_difficulty(self):
        """Similar colors should produce high game difficulty."""
        from unittest.mock import patch
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        frame[:, :200] = [130, 130, 130]  # Dark gray
        frame[:, 200:] = [140, 140, 140]  # Slightly lighter gray
        bboxes = [[50, 20, 150, 180], [250, 20, 350, 180]]

        clf = VLMTeamClassifier()
        # Mock model loads since torch not available in test env
        with patch.object(clf, '_ensure_siglip'), \
             patch.object(clf, '_build_team_profiles'), \
             patch.object(clf, '_ensure_clip'), \
             patch.object(clf, '_build_clip_profiles'):
            clf.fit(frame, bboxes)

        assert clf.get_game_difficulty() > 0.7
        assert clf.get_cluster_separation() < CLUSTER_SEPARATION_MIN


# Import here to avoid issues if config changes
from src.config import CLUSTER_SEPARATION_MIN
