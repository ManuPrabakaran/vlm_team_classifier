import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.classifier import VLMTeamClassifier, _bbox_center
from src.config import (
    KMEANS_CONFIDENCE_THRESHOLD,
    SIGLIP_CONFIDENCE_THRESHOLD,
    CLUSTER_SEPARATION_MIN,
    OUTPUT_METHOD_KMEANS,
    OUTPUT_METHOD_VLM,
    OUTPUT_METHOD_MANUAL,
)


@pytest.fixture
def two_color_frame():
    """200x400 frame: left half red, right half blue."""
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    frame[:, :200] = [0, 0, 255]   # Red (BGR)
    frame[:, 200:] = [255, 0, 0]   # Blue (BGR)
    return frame


@pytest.fixture
def two_color_bboxes():
    return [
        [50, 20, 150, 180],   # Red region
        [250, 20, 350, 180],  # Blue region
    ]


@pytest.fixture
def classifier():
    return VLMTeamClassifier()


@pytest.fixture
def no_ablation_classifier():
    """Classifier with all ablation flags off — pure cascade."""
    return VLMTeamClassifier(config={
        "ablation": {
            "court_position": False,
            "number_lookup": False,
            "clip_ensemble": False,
            "temporal_consistency": False,
        }
    })


@pytest.fixture
def fitted_classifier(two_color_frame, two_color_bboxes):
    """Classifier fitted on clearly distinct colors (high separation)."""
    clf = VLMTeamClassifier()
    clf.fit(two_color_frame, two_color_bboxes)
    return clf


# ── Init ──

class TestInit:
    def test_default_thresholds(self, classifier):
        assert classifier.kmeans_threshold == KMEANS_CONFIDENCE_THRESHOLD
        assert classifier.siglip_threshold == SIGLIP_CONFIDENCE_THRESHOLD

    def test_custom_config(self):
        clf = VLMTeamClassifier(config={"kmeans_threshold": 0.9})
        assert clf.kmeans_threshold == 0.9

    def test_cascade_stats_zeroed(self, classifier):
        stats = classifier.get_cascade_stats()
        assert stats["kmeans"] == 0
        assert stats["temporal"] == 0
        assert stats["number_lookup"] == 0

    def test_models_not_loaded(self, classifier):
        assert classifier._siglip_model is None
        assert classifier._clip_model is None
        assert classifier._qwen_model is None

    def test_ablation_flags_loaded(self, classifier):
        assert "court_position" in classifier.ablation
        assert "temporal_consistency" in classifier.ablation


# ── Game Context ──

class TestSetGameContext:
    def test_stores_team_names(self, classifier):
        classifier.set_game_context(team_names={0: "Celtics", 1: "Heat"})
        assert classifier._team_names == {0: "Celtics", 1: "Heat"}

    def test_stores_jersey_descriptions(self, classifier):
        descs = {0: "white jersey", 1: "black jersey"}
        classifier.set_game_context(jersey_descriptions=descs)
        assert classifier._jersey_descriptions == descs

    def test_special_jersey_defaults_empty(self, classifier):
        classifier.set_game_context()
        assert classifier._special_jersey == {}


# ── Roster / Number Lookup ──

class TestRoster:
    def test_set_roster_stores_numbers(self, classifier):
        classifier.set_roster([1, 5, 10, 23], [2, 7, 11, 30])
        assert "23" in classifier._roster[0]
        assert "30" in classifier._roster[1]

    def test_unique_numbers_detected(self, classifier):
        classifier.set_roster([1, 5, 10, 23], [2, 7, 11, 30])
        # All numbers are unique since no overlap
        assert classifier._unique_numbers["23"] == 0
        assert classifier._unique_numbers["30"] == 1

    def test_shared_numbers_excluded(self, classifier):
        classifier.set_roster([1, 5, 10], [5, 7, 11])
        # Number 5 is on both teams — not unique
        assert "5" not in classifier._unique_numbers
        assert "1" in classifier._unique_numbers
        assert classifier._unique_numbers["1"] == 0

    def test_check_jersey_number_found(self, classifier):
        classifier.set_roster([23], [30])
        team, conf = classifier._check_jersey_number("23")
        assert team == 0
        assert conf == 0.99

    def test_check_jersey_number_not_found(self, classifier):
        classifier.set_roster([23], [30])
        team, conf = classifier._check_jersey_number("99")
        assert team is None

    def test_check_jersey_number_disabled(self, no_ablation_classifier):
        no_ablation_classifier.set_roster([23], [30])
        team, conf = no_ablation_classifier._check_jersey_number("23")
        assert team is None

    def test_predict_with_jersey_number(self, fitted_classifier, two_color_frame, two_color_bboxes):
        fitted_classifier.set_roster([23], [30])
        result = fitted_classifier.predict(
            two_color_frame, two_color_bboxes[0], jersey_number="23"
        )
        assert result["team_id"] == 0
        assert result["confidence"] == 0.99
        assert result["method"] == "number_lookup"

    def test_number_lookup_stat_counted(self, fitted_classifier, two_color_frame, two_color_bboxes):
        fitted_classifier.set_roster([23], [30])
        fitted_classifier.predict(two_color_frame, two_color_bboxes[0], jersey_number="23")
        stats = fitted_classifier.get_cascade_stats()
        assert stats["number_lookup"] == 1


# ── Court Position ──

class TestCourtPosition:
    def test_returns_none_with_few_players(self, classifier):
        team, conf = classifier._court_position_prior([50, 50, 100, 100], [[50, 50, 100, 100]])
        assert team is None

    def test_returns_team_with_spread_players(self, classifier):
        # 6 players spread across x=50..950
        all_bboxes = [
            [50, 50, 100, 200], [100, 50, 150, 200], [150, 50, 200, 200],
            [700, 50, 750, 200], [800, 50, 850, 200], [900, 50, 950, 200],
        ]
        # Player far left should get team 0
        team, conf = classifier._court_position_prior([50, 50, 100, 200], all_bboxes)
        assert team == 0
        assert 0.55 <= conf <= 0.65

        # Player far right should get team 1
        team, conf = classifier._court_position_prior([900, 50, 950, 200], all_bboxes)
        assert team == 1

    def test_returns_none_when_bunched(self, classifier):
        # All players within 80px — too bunched for a signal
        all_bboxes = [
            [100, 50, 120, 200], [110, 50, 130, 200],
            [120, 50, 140, 200], [130, 50, 150, 200],
        ]
        team, conf = classifier._court_position_prior([100, 50, 120, 200], all_bboxes)
        assert team is None

    def test_disabled_by_ablation(self, no_ablation_classifier):
        all_bboxes = [
            [50, 50, 100, 200], [100, 50, 150, 200],
            [700, 50, 750, 200], [800, 50, 850, 200],
        ]
        team, conf = no_ablation_classifier._court_position_prior([50, 50, 100, 200], all_bboxes)
        assert team is None

    def test_court_position_in_signals(self, fitted_classifier, two_color_frame):
        # Wide-spread bboxes so court position fires
        bboxes = [
            [10, 20, 60, 180], [30, 20, 80, 180],
            [300, 20, 350, 180], [350, 20, 395, 180],
        ]
        fitted_classifier.fit(two_color_frame, bboxes)
        result = fitted_classifier.predict(two_color_frame, bboxes[0], all_bboxes=bboxes)
        # Court position should appear in signals if spread is enough
        if "court_position" in result["signals"]:
            assert result["signals"]["court_position"]["confidence"] <= 0.65


# ── Temporal Consistency ──

class TestTemporalConsistency:
    def test_cache_miss_on_first_predict(self, classifier):
        team, conf = classifier._check_temporal_cache([100, 100, 200, 200])
        assert team is None

    def test_cache_hit_after_update(self, classifier):
        bbox = [100, 100, 200, 200]
        classifier._update_temporal_cache(bbox, 0, 0.92)
        team, conf = classifier._check_temporal_cache(bbox)
        assert team == 0
        assert conf >= 0.90

    def test_cache_hit_with_position_tolerance(self, classifier):
        classifier._update_temporal_cache([100, 100, 200, 200], 1, 0.88)
        # Slightly shifted bbox should still match
        team, conf = classifier._check_temporal_cache([120, 110, 220, 210])
        assert team == 1

    def test_cache_miss_with_large_shift(self, classifier):
        classifier._update_temporal_cache([100, 100, 200, 200], 1, 0.88)
        # Far away bbox should not match
        team, conf = classifier._check_temporal_cache([500, 500, 600, 600])
        assert team is None

    def test_cache_expires_after_interval(self, classifier):
        classifier._update_temporal_cache([100, 100, 200, 200], 0, 0.92)
        # Advance past reidentification interval
        classifier._frame_number = 50
        team, conf = classifier._check_temporal_cache([100, 100, 200, 200])
        assert team is None

    def test_confidence_decays_over_frames(self, classifier):
        classifier._update_temporal_cache([100, 100, 200, 200], 0, 0.90)
        classifier._frame_number = 5
        team, conf = classifier._check_temporal_cache([100, 100, 200, 200])
        assert team == 0
        assert conf < 0.90  # Decayed

    def test_temporal_resolves_before_kmeans(self, fitted_classifier, two_color_frame, two_color_bboxes):
        # First predict populates cache
        fitted_classifier.predict(two_color_frame, two_color_bboxes[0])
        # Second predict at same position should use temporal cache
        result = fitted_classifier.predict(two_color_frame, two_color_bboxes[0])
        assert result["method"] == "temporal"
        assert fitted_classifier.get_cascade_stats()["temporal"] >= 1

    def test_disabled_by_ablation(self, no_ablation_classifier, two_color_frame, two_color_bboxes):
        no_ablation_classifier.fit(two_color_frame, two_color_bboxes)
        no_ablation_classifier.predict(two_color_frame, two_color_bboxes[0])
        # Second predict should NOT use temporal (disabled)
        result = no_ablation_classifier.predict(two_color_frame, two_color_bboxes[0])
        assert result["method"] != "temporal"

    def test_advance_frame(self, classifier):
        classifier.advance_frame()
        assert classifier._frame_number == 1

    def test_get_track_cache_size(self, classifier):
        assert classifier.get_track_cache_size() == 0
        classifier._update_temporal_cache([100, 100, 200, 200], 0, 0.9)
        assert classifier.get_track_cache_size() == 1


# ── Game Difficulty ──

class TestGameDifficulty:
    def test_high_separation_low_difficulty(self, fitted_classifier):
        assert fitted_classifier.get_game_difficulty() < 0.3

    def test_default_when_no_separation(self, classifier):
        classifier._compute_game_difficulty()
        assert classifier.get_game_difficulty() == 0.5

    def test_special_jersey_bumps_difficulty(self, fitted_classifier):
        base = fitted_classifier.get_game_difficulty()
        fitted_classifier._special_jersey = {0: True}
        fitted_classifier._compute_game_difficulty()
        assert fitted_classifier.get_game_difficulty() >= base


# ── Cluster Separation ──

class TestClusterSeparation:
    def test_high_separation_passes(self, fitted_classifier):
        assert fitted_classifier._check_cluster_separation() is True
        assert fitted_classifier.get_cluster_separation() > CLUSTER_SEPARATION_MIN

    def test_none_separation_fails(self, classifier):
        assert classifier._check_cluster_separation() is False


# ── Fit ──

class TestFit:
    def test_sets_cluster_separation(self, fitted_classifier):
        assert fitted_classifier.get_cluster_separation() is not None

    def test_sets_game_difficulty(self, fitted_classifier):
        assert fitted_classifier.get_game_difficulty() is not None

    def test_no_siglip_when_separation_high(self, fitted_classifier):
        assert fitted_classifier._siglip_model is None
        assert fitted_classifier._team_profiles == {}


# ── Predict K-Means ──

class TestPredictKmeans:
    def test_returns_team_and_confidence(self, fitted_classifier, two_color_frame, two_color_bboxes):
        team_id, conf = fitted_classifier._predict_kmeans(two_color_frame, two_color_bboxes[0])
        assert team_id in (0, 1)
        assert 0.5 <= conf <= 1.0

    def test_distinct_colors_high_confidence(self, fitted_classifier, two_color_frame, two_color_bboxes):
        _, conf = fitted_classifier._predict_kmeans(two_color_frame, two_color_bboxes[0])
        assert conf > 0.9


# ── Predict (full cascade) ──

class TestPredict:
    def test_returns_correct_structure(self, fitted_classifier, two_color_frame, two_color_bboxes):
        result = fitted_classifier.predict(two_color_frame, two_color_bboxes[0])
        assert "team_id" in result
        assert "confidence" in result
        assert "method" in result
        assert "signals" in result

    def test_high_separation_resolves_at_kmeans(self, fitted_classifier, two_color_frame, two_color_bboxes):
        result = fitted_classifier.predict(two_color_frame, two_color_bboxes[0])
        assert result["method"] == OUTPUT_METHOD_KMEANS

    def test_team_id_is_int(self, fitted_classifier, two_color_frame, two_color_bboxes):
        result = fitted_classifier.predict(two_color_frame, two_color_bboxes[0])
        assert isinstance(result["team_id"], int)

    def test_different_teams_for_different_colors(self, fitted_classifier, two_color_frame, two_color_bboxes):
        # Disable temporal so second predict doesn't use cache from first
        fitted_classifier.ablation["temporal_consistency"] = False
        r0 = fitted_classifier.predict(two_color_frame, two_color_bboxes[0])
        r1 = fitted_classifier.predict(two_color_frame, two_color_bboxes[1])
        assert r0["team_id"] != r1["team_id"]


# ── Predict Batch ──

class TestPredictBatch:
    def test_returns_list(self, fitted_classifier, two_color_frame, two_color_bboxes):
        fitted_classifier.ablation["temporal_consistency"] = False
        results = fitted_classifier.predict_batch(two_color_frame, two_color_bboxes)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_each_result_has_structure(self, fitted_classifier, two_color_frame, two_color_bboxes):
        fitted_classifier.ablation["temporal_consistency"] = False
        results = fitted_classifier.predict_batch(two_color_frame, two_color_bboxes)
        for r in results:
            assert "team_id" in r
            assert "method" in r

    def test_batch_with_jersey_numbers(self, fitted_classifier, two_color_frame, two_color_bboxes):
        fitted_classifier.set_roster([23], [30])
        results = fitted_classifier.predict_batch(
            two_color_frame, two_color_bboxes, jersey_numbers=["23", "30"]
        )
        assert results[0]["method"] == "number_lookup"
        assert results[0]["team_id"] == 0
        assert results[1]["method"] == "number_lookup"
        assert results[1]["team_id"] == 1


# ── Cascade Escalation ──

class TestCascadeEscalation:
    def test_low_confidence_escalates_to_siglip(self, two_color_frame, two_color_bboxes):
        clf = VLMTeamClassifier(config={
            "kmeans_threshold": 1.0,
            "separation_min": 9999.0,
            "ablation": {"court_position": False, "number_lookup": False,
                         "clip_ensemble": False, "temporal_consistency": False},
        })
        with patch.object(clf, '_ensure_siglip'):
            with patch.object(clf, '_build_team_profiles'):
                clf.fit(two_color_frame, two_color_bboxes)

        fake_emb = np.random.randn(768).astype(np.float32)
        fake_emb = fake_emb / np.linalg.norm(fake_emb)
        clf._team_profiles = {0: fake_emb, 1: -fake_emb}

        with patch.object(clf, '_ensure_siglip'):
            with patch.object(clf, '_predict_siglip', return_value=(0, 0.95)):
                result = clf.predict(two_color_frame, two_color_bboxes[0])
                assert "siglip" in result["method"]

    def test_all_low_falls_to_manual(self, two_color_frame, two_color_bboxes):
        clf = VLMTeamClassifier(config={
            "kmeans_threshold": 1.0,
            "siglip_threshold": 1.0,
            "separation_min": 9999.0,
            "ablation": {"court_position": False, "number_lookup": False,
                         "clip_ensemble": False, "temporal_consistency": False},
        })
        with patch.object(clf, '_ensure_siglip'):
            with patch.object(clf, '_build_team_profiles'):
                clf.fit(two_color_frame, two_color_bboxes)
        clf._team_profiles = {0: np.ones(768), 1: -np.ones(768)}

        with patch.object(clf, '_ensure_siglip'):
            with patch.object(clf, '_predict_siglip', return_value=(0, 0.5)):
                result = clf.predict(two_color_frame, two_color_bboxes[0])
                assert result["method"] == OUTPUT_METHOD_MANUAL


# ── Cascade Stats ──

class TestCascadeStats:
    def test_kmeans_counted(self, fitted_classifier, two_color_frame, two_color_bboxes):
        fitted_classifier.predict(two_color_frame, two_color_bboxes[0])
        stats = fitted_classifier.get_cascade_stats()
        assert stats["kmeans"] == 1

    def test_batch_increments(self, fitted_classifier, two_color_frame, two_color_bboxes):
        fitted_classifier.ablation["temporal_consistency"] = False
        fitted_classifier.predict_batch(two_color_frame, two_color_bboxes)
        stats = fitted_classifier.get_cascade_stats()
        assert stats["kmeans"] == 2


# ── Composite Confidence ──

class TestCompositeConfidence:
    def test_single_signal(self, classifier):
        signals = {"kmeans": {"team_id": 0, "confidence": 0.9}}
        conf = classifier._compute_composite_confidence(signals)
        assert conf == pytest.approx(0.9, abs=0.01)

    def test_no_signals(self, classifier):
        assert classifier._compute_composite_confidence({}) == 0.5

    def test_weighted_average(self, classifier):
        signals = {
            "kmeans": {"team_id": 0, "confidence": 0.8},
            "siglip": {"team_id": 0, "confidence": 1.0},
        }
        conf = classifier._compute_composite_confidence(signals)
        # Weighted: (0.20*0.8 + 0.30*1.0) / (0.20+0.30) = 0.46/0.50 = 0.92
        assert conf == pytest.approx(0.92, abs=0.01)


# ── Bbox Center Helper ──

class TestBboxCenter:
    def test_center_calculation(self):
        assert _bbox_center([0, 0, 100, 200]) == (50.0, 100.0)
        assert _bbox_center([100, 50, 300, 150]) == (200.0, 100.0)
