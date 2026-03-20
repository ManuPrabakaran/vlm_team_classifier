import numpy as np
import pytest
from src.baseline import TeamClustering


@pytest.fixture
def clustering():
    return TeamClustering(n_teams=2)


@pytest.fixture
def two_color_frame():
    """Create a 200x400 frame: left half red, right half blue."""
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    frame[:, :200] = [0, 0, 255]   # Red (BGR)
    frame[:, 200:] = [255, 0, 0]   # Blue (BGR)
    return frame


@pytest.fixture
def two_color_bboxes():
    """Bboxes centered in left (red) and right (blue) halves."""
    return [
        [50, 20, 150, 180],   # Red region
        [250, 20, 350, 180],  # Blue region
    ]


class TestExtractJerseyColor:
    def test_returns_3d_color(self, clustering, two_color_frame, two_color_bboxes):
        color = clustering.extract_jersey_color(two_color_frame, two_color_bboxes[0])
        assert color.shape == (3,)

    def test_red_region_is_red(self, clustering, two_color_frame, two_color_bboxes):
        color = clustering.extract_jersey_color(two_color_frame, two_color_bboxes[0])
        # BGR: red channel (index 2) should dominate
        assert color[2] > 200
        assert color[0] < 50
        assert color[1] < 50

    def test_blue_region_is_blue(self, clustering, two_color_frame, two_color_bboxes):
        color = clustering.extract_jersey_color(two_color_frame, two_color_bboxes[1])
        # BGR: blue channel (index 0) should dominate
        assert color[0] > 200
        assert color[2] < 50


class TestFit:
    def test_creates_kmeans(self, clustering, two_color_frame, two_color_bboxes):
        clustering.fit(two_color_frame, two_color_bboxes)
        assert clustering.kmeans is not None

    def test_two_clusters(self, clustering, two_color_frame, two_color_bboxes):
        clustering.fit(two_color_frame, two_color_bboxes)
        assert clustering.kmeans.n_clusters == 2

    def test_centroids_are_distinct(self, clustering, two_color_frame, two_color_bboxes):
        clustering.fit(two_color_frame, two_color_bboxes)
        c0, c1 = clustering.kmeans.cluster_centers_
        distance = np.linalg.norm(c0 - c1)
        assert distance > 100  # Red and blue are far apart


class TestPredictTeam:
    def test_returns_0_or_1(self, clustering, two_color_frame, two_color_bboxes):
        clustering.fit(two_color_frame, two_color_bboxes)
        pred = clustering.predict_team(two_color_frame, two_color_bboxes[0])
        assert pred in (0, 1)

    def test_different_colors_get_different_teams(self, clustering, two_color_frame, two_color_bboxes):
        clustering.fit(two_color_frame, two_color_bboxes)
        pred0 = clustering.predict_team(two_color_frame, two_color_bboxes[0])
        pred1 = clustering.predict_team(two_color_frame, two_color_bboxes[1])
        assert pred0 != pred1

    def test_same_color_gets_same_team(self, clustering, two_color_frame):
        bboxes = [[50, 20, 150, 180], [250, 20, 350, 180]]
        clustering.fit(two_color_frame, bboxes)
        # Two bboxes in the same red region
        bbox_a = [50, 20, 100, 180]
        bbox_b = [60, 20, 140, 180]
        pred_a = clustering.predict_team(two_color_frame, bbox_a)
        pred_b = clustering.predict_team(two_color_frame, bbox_b)
        assert pred_a == pred_b
