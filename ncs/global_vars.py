import os

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BODY_DIR = os.path.join(ROOT_DIR, "body_models")
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")
DATA_DIR = os.path.join(ROOT_DIR, "data")
TXT_DIR = os.path.join(ROOT_DIR, "ncs", "dataset", "txt")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
SMPL_DIR = os.path.join(ROOT_DIR, "smpl")
TMP_DIR = os.path.join(ROOT_DIR, "tmp")

# Skeletons
NUM_JOINTS = {"smpl": 24, "mixamo": 65}

# Physick
GRAVITY = 9.81
