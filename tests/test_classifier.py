import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.classifier import VLMTeamClassifier
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
def fitted_classifier(two_color_frame, two_color_bboxes):
    """Classifier fitted on clearly distinct colors (high separation)."""
    clf = VLMTeamClassifier()
    clf.fit(two_color_frame, two_color_bboxes)
    return clf


class TestInit:
    def test_default_thresholds(self, classifier):
        assert classifier.kmeans_threshold == KMEANS_CONFIDENCE_THRESHOLD
        assert classifier.siglip_threshold == SIGLIP_CONFIDENCE_THRESHOLD

    def test_custom_config(self):
        clf = VLMTeamClassifier(config={"kmeans_threshold": 0.9})
        assert clf.kmeans_threshold == 0.9

    def test_cascade_stats_zeroed(self, classifier):
        stats = classifier.get_cascade_stats()
        assert all(v == 0 for v in stats.values())

    def test_models_not_loaded(self, classifier):
        assert classifier._siglip_model is None
        assert classifier._qwen_model is None


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


class TestGameDifficulty:
    def test_high_separation_low_difficulty(self, fitted_classifier):
        # Red vs blue → separation > 200 → difficulty near 0
        assert fitted_classifier.get_game_difficulty() < 0.3

    def test_default_when_no_separation(self, classifier):
        classifier._compute_game_difficulty()
        assert classifier.get_game_difficulty() == 0.5

    def test_special_jersey_bumps_difficulty(self, fitted_classifier):
        base = fitted_classifier.get_game_difficulty()
        fitted_classifier._special_jersey = {0: True}
        fitted_classifier._compute_game_difficulty()
        assert fitted_classifier.get_game_difficulty() >= base


class TestClusterSeparation:
    def test_high_separation_passes(self, fitted_classifier):
        assert fitted_classifier._check_cluster_separation() is True
        assert fitted_classifier.get_cluster_separation() > CLUSTER_SEPARATION_MIN

    def test_none_separation_fails(self, classifier):
        assert classifier._check_cluster_separation() is False


class TestCourtPositionPrior:
    def test_stub_returns_none(self, classifier):
        team, conf = classifier._court_position_prior([0, 0, 10, 10], [])
        assert team is None
        assert conf == 0.0


class TestFit:
    def test_sets_cluster_separation(self, fitted_classifier):
        assert fitted_classifier.get_cluster_separation() is not None

    def test_sets_game_difficulty(self, fitted_classifier):
        assert fitted_classifier.get_game_difficulty() is not None

    def test_no_siglip_when_separation_high(self, fitted_classifier):
        # High separation → SigLIP not loaded, no team profiles
        assert fitted_classifier._siglip_model is None
        assert fitted_classifier._team_profiles == {}


class TestPredictKmeans:
    def test_returns_team_and_confidence(self, fitted_classifier, two_color_frame, two_color_bboxes):
        team_id, conf = fitted_classifier._predict_kmeans(two_color_frame, two_color_bboxes[0])
        assert team_id in (0, 1)
        assert 0.5 <= conf <= 1.0

    def test_distinct_colors_high_confidence(self, fitted_classifier, two_color_frame, two_color_bboxes):
        _, conf = fitted_classifier._predict_kmeans(two_color_frame, two_color_bboxes[0])
        assert conf > 0.9  # Red vs blue should be very confident


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
        r0 = fitted_classifier.predict(two_color_frame, two_color_bboxes[0])
        r1 = fitted_classifier.predict(two_color_frame, two_color_bboxes[1])
        assert r0["team_id"] != r1["team_id"]


class TestPredictBatch:
    def test_returns_list(self, fitted_classifier, two_color_frame, two_color_bboxes):
        results = fitted_classifier.predict_batch(two_color_frame, two_color_bboxes)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_each_result_has_structure(self, fitted_classifier, two_color_frame, two_color_bboxes):
        results = fitted_classifier.predict_batch(two_color_frame, two_color_bboxes)
        for r in results:
            assert "team_id" in r
            assert "method" in r


class TestCascadeEscalation:
    def test_low_confidence_escalates_to_siglip(self, two_color_frame, two_color_bboxes):
        """When K-Means separation is low, should try SigLIP."""
        clf = VLMTeamClassifier(config={
            "kmeans_threshold": 1.0,
            "separation_min": 9999.0,  # Force separation check to fail
        })
        with patch.object(clf, '_ensure_siglip'):
            with patch.object(clf, '_build_team_profiles'):
                clf.fit(two_color_frame, two_color_bboxes)

        # Mock SigLIP to avoid loading real model
        fake_emb = np.random.randn(768).astype(np.float32)
        fake_emb = fake_emb / np.linalg.norm(fake_emb)
        clf._team_profiles = {0: fake_emb, 1: -fake_emb}

        with patch.object(clf, '_ensure_siglip'):
            with patch.object(clf, '_predict_siglip', return_value=(0, 0.95)):
                result = clf.predict(two_color_frame, two_color_bboxes[0])
                assert "siglip" in result["method"]

    def test_all_low_falls_to_manual(self, two_color_frame, two_color_bboxes):
        """When all stages have low confidence, should fall to manual."""
        clf = VLMTeamClassifier(config={
            "kmeans_threshold": 1.0,
            "siglip_threshold": 1.0,
            "separation_min": 9999.0,
        })
        with patch.object(clf, '_ensure_siglip'):
            with patch.object(clf, '_build_team_profiles'):
                clf.fit(two_color_frame, two_color_bboxes)
        clf._team_profiles = {0: np.ones(768), 1: -np.ones(768)}

        with patch.object(clf, '_ensure_siglip'):
            with patch.object(clf, '_predict_siglip', return_value=(0, 0.5)):
                result = clf.predict(two_color_frame, two_color_bboxes[0])
                assert result["method"] == OUTPUT_METHOD_MANUAL


class TestCascadeStats:
    def test_kmeans_counted(self, fitted_classifier, two_color_frame, two_color_bboxes):
        fitted_classifier.predict(two_color_frame, two_color_bboxes[0])
        stats = fitted_classifier.get_cascade_stats()
        assert stats["kmeans"] == 1

    def test_batch_increments(self, fitted_classifier, two_color_frame, two_color_bboxes):
        fitted_classifier.predict_batch(two_color_frame, two_color_bboxes)
        stats = fitted_classifier.get_cascade_stats()
        assert stats["kmeans"] == 2


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
