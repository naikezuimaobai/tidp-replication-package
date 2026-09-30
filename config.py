"""Configuration for TIDP Framework - Linux Version"""

import os

# ============ Base directory ============
BASE_DIR = os.environ.get("TIDP_BASE_DIR", "/model/bd_gfk/TCDDP/icddp")

# ============ Model paths ============
SENTENCE_BERT_MODEL_PATH = os.environ.get(
    "TIDP_SENTENCE_BERT_MODEL", "/model/bd_gfk/TCDDP/models/all-MiniLM-L6-v2"
)

# ============ Dataset path ============
DATASET_PATH = os.environ.get(
    "TIDP_DATASET_PATH", "/model/bd_gfk/TCDDP/dataset/Mind2Web"
)

# ============ LLM configuration ============
LLM_API_BASE = os.environ.get("TIDP_LLM_API_BASE", "http://localhost:11434/v1")
LLM_MODEL = os.environ.get("TIDP_LLM_MODEL", "qwen2.5:3b")
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0

# ============ GPU configuration ============
# Do not hard-code physical GPU IDs. Set CUDA_VISIBLE_DEVICES before launching
# the program, or use TIDP_CUDA_VISIBLE_DEVICES for a package-local override.
CUDA_VISIBLE_DEVICES = os.environ.get("TIDP_CUDA_VISIBLE_DEVICES", "")
if CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

# ============ Dynamic threshold parameters ============
N_MIN = 20
N_MAX = 500

INTENT_BASE_PARAMS = {
    'content_browsing': {
        'high': 0.60, 'low': 0.40, 'p1': 0.10, 'p2': 0.45  # high: 0.55→0.60
    },
    'form_selection': {
        'high': 0.60, 'low': 0.40, 'p1': 0.10, 'p2': 0.45  # unchanged
    },
    'information_input': {
        'high': 0.60, 'low': 0.40, 'p1': 0.10, 'p2': 0.45  # unchanged
    },
    'button_interaction': {
        'high': 0.55, 'low': 0.30, 'p1': 0.10, 'p2': 0.50  # high: 0.50→0.55
    }
}
# INTENT_BASE_PARAMS = {
#     'content_browsing': {
#         'high': 0.55, 'low': 0.35, 'p1': 0.15, 'p2': 0.50  # higher threshold
#     },
#     'form_selection': {
#         'high': 0.60, 'low': 0.40, 'p1': 0.15, 'p2': 0.50
#     },
#     'information_input': {
#         'high': 0.60, 'low': 0.40, 'p1': 0.15, 'p2': 0.45
#     },
#     'button_interaction': {
#         'high': 0.50, 'low': 0.25, 'p1': 0.15, 'p2': 0.55
#     }
# }


# ============ Context limit ============
CONTEXT_LIMIT_TOKENS = 8000
USABLE_CONTEXT_RATIO = 0.8  # Historical value retained for compatibility.
AVG_TOKENS_PER_ELEMENT = 150

# ============ Fallback parameters ============
FALLBACK_MIN_ELEMENTS = 5

# ============ Proximity parameters ============
MAX_DOM_DEPTH_DIFF = 3
EDGE_THRESHOLD_RATIO = 0.15

# ============ DOM text extraction ============
MAX_ELEMENT_TEXT_LEN = 200

# ============ Intent mapping ============
OP_TO_INTENT = {
    'CLICK': 'button_interaction',
    'TYPE': 'information_input',
    'SELECT': 'form_selection'
}
