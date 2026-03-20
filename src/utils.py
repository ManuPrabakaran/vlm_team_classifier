import subprocess
import numpy as np
import cv2
from src.config import (
    YOLO_CONFIDENCE,
    YOLO_PERSON_CLASS_ID,
    MODEL_SIGLIP,
    MODEL_QWEN,
)

import torch

def get_device():
    """Return 'cuda' if GPU available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def download_youtube_clip(url, output_path, start_time=0, duration=30):
    """Download a section of a YouTube video using yt-dlp."""
    cmd = [
        "yt-dlp",
        "--download-sections", f"*{start_time}-{start_time + duration}",
        "--force-keyframes-at-cuts",
        "-o", output_path,
        url,
    ]
    subprocess.run(cmd, check=True)


def extract_frames(video_path, sample_rate=1):
    """
    Extract frames from a video at the given sample rate (frames per second).
    Returns a list of (frame, timestamp_seconds) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / sample_rate)

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            timestamp = frame_idx / fps
            frames.append((frame, timestamp))
        frame_idx += 1

    cap.release()
    return frames


def detect_players(frame, model, confidence=YOLO_CONFIDENCE):
    """
    Run YOLOv8 on a frame and return bounding boxes for detected people.
    Returns a list of [x1, y1, x2, y2] bboxes.
    """
    results = model(frame, conf=confidence)[0]
    bboxes = []
    for box in results.boxes:
        if int(box.cls[0]) == YOLO_PERSON_CLASS_ID:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bboxes.append([x1, y1, x2, y2])
    return bboxes


def crop_player(frame, bbox):
    """Extract a player crop from a frame given a [x1, y1, x2, y2] bbox."""
    x1, y1, x2, y2 = bbox
    return frame[y1:y2, x1:x2]


def compute_cluster_separation(kmeans_model):
    """
    Compute Euclidean distance between K-Means centroids in RGB space.
    Low separation → teams have similar jersey colors → escalate to VLM.
    """
    c0, c1 = kmeans_model.cluster_centers_
    return float(np.linalg.norm(c0 - c1))


def compute_centroid_distance_confidence(color, kmeans_model):
    """
    Derive classification confidence from centroid distance ratio.
    Confidence = distance to losing centroid / (sum of both distances).
    Returns a float in (0.5, 1.0].
    """
    centers = kmeans_model.cluster_centers_
    d0 = np.linalg.norm(color - centers[0])
    d1 = np.linalg.norm(color - centers[1])
    total = d0 + d1
    if total == 0:
        return 0.5
    predicted = 0 if d0 < d1 else 1
    confidence = (d1 if predicted == 0 else d0) / total
    return float(confidence)


def load_siglip_model(device=None):
    """Load SigLIP model and processor from HuggingFace."""
    device = device or get_device()
    from transformers import AutoProcessor, AutoModel
    processor = AutoProcessor.from_pretrained(MODEL_SIGLIP)
    model = AutoModel.from_pretrained(MODEL_SIGLIP).to(device)
    model.eval()
    return model, processor


def load_qwen_model(device=None):
    """Load Qwen2-VL model and processor from HuggingFace."""
    device = device or get_device()
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_QWEN)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_QWEN, torch_dtype="auto"
    ).to(device)
    model.eval()
    return model, processor


def extract_siglip_embedding(crop, model, processor, device=None):
    """
    Extract a SigLIP visual embedding from a player crop.
    Returns a 1D numpy array of shape (embedding_dim,).
    """
    device = device or get_device()
    from PIL import Image
    image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    emb = outputs[0].cpu()
    if emb.dim() == 2:  # (num_patches, hidden) — mean pool to get 1D vector
        emb = emb.mean(dim=0)
    embedding = emb.numpy()
    return embedding / np.linalg.norm(embedding)  # L2 normalize


def compute_embedding_distance(embedding, team_profiles):
    """
    Compute cosine distance from an embedding to each team profile centroid.
    team_profiles: dict {0: np.array, 1: np.array} of L2-normalized centroids.
    Returns dict {0: distance, 1: distance}.
    """
    distances = {}
    for team_id, centroid in team_profiles.items():
        distances[team_id] = float(1 - np.dot(embedding, centroid))
    return distances

def draw_bboxes(frame, bboxes, labels=None, color=(0, 255, 0)):
    """Draw bounding boxes on a frame for visualization."""
    frame_copy = frame.copy()
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
        if labels:
            cv2.putText(frame_copy, str(labels[i]), (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame_copy