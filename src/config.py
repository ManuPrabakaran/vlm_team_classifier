# Model identifiers
MODEL_SIGLIP = "google/siglip-base-patch16-224"
MODEL_CLIP = "openai/clip-vit-base-patch32"
MODEL_QWEN = "Qwen/Qwen2-VL-2B-Instruct"
MODEL_SMOLVLM = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"  # Research comparison only — already in Paloa Labs pipeline
MODEL_YOLO = "yolov8n.pt"

# Cascade confidence thresholds
KMEANS_CONFIDENCE_THRESHOLD = 0.85
SIGLIP_CONFIDENCE_THRESHOLD = 0.80
QWEN_CONFIDENCE_THRESHOLD = 0.75
CLUSTER_SEPARATION_MIN = 30.0  # Min RGB distance between K-Means centroids before forcing VLM

# Team constants
N_TEAMS = 2
REFEREE_TEAM_ID = -1

# Composite confidence weights — only signals that are actually computed.
# Weights renormalized over active signals at runtime, so they don't need to sum to 1.0.
COMPOSITE_WEIGHTS = {
    "kmeans":         0.20,
    "siglip":         0.30,
    "clip":           0.15,
    "court_position": 0.00,  # Infrastructure only — effective weight requires player-position tags (future work)
    "number_lookup":  0.25,
}

# Ablation flags — toggle individual signals on/off for ablation studies.
# Set to False to disable a signal without removing code.
ABLATION_FLAGS = {
    "court_position": True,
    "number_lookup": True,
    "clip_ensemble": True,
    "temporal_consistency": True,
}

# Game difficulty constant thresholds
GAME_DIFFICULTY_HIGH_THRESHOLD = 0.7   # Above this → aggressive VLM routing
GAME_DIFFICULTY_LOW_THRESHOLD = 0.3    # Below this → trust K-Means heavily

# Temporal consistency
TEMPORAL_POSITION_TOLERANCE = 50  # pixels — max bbox center drift to consider same player
TEMPORAL_MIN_CONFIDENCE = 0.80    # minimum confidence to lock a player's team assignment
REIDENTIFICATION_INTERVAL = 30    # Re-classify locked players every N frames

# Frame sampling
FRAME_SAMPLE_RATE_FPS = 1         # Frames per second to extract for analysis
TIPOFF_FRAMES = 5                 # Number of tipoff frames used to build team profiles

# Batch processing
BATCH_SIZE_SIGLIP = 10            # Max player crops per SigLIP forward pass

# YOLO detection
YOLO_CONFIDENCE = 0.5
YOLO_PERSON_CLASS_ID = 0

# Output defaults
OUTPUT_METHOD_KMEANS = "kmeans"
OUTPUT_METHOD_VLM = "vlm"
OUTPUT_METHOD_MANUAL = "manual"
