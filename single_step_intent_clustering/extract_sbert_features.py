# ===================== extract_sbert_features.py =====================
# Extract SBERT features from preprocessed single-step operations.
# Input: step_actions_simplified.xlsx.
# Output: sbert_features.npy and sbert_features_normalized.npy.
# ====================================================================

import numpy as np
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import time
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
BASE_PATH = os.environ.get("TIDP_CLUSTER_BASE_PATH", os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = f"{BASE_PATH}\\step_actions_simplified.xlsx"
OUTPUT_FEATURES = f"{BASE_PATH}\\sbert_features.npy"
OUTPUT_FEATURES_NORM = f"{BASE_PATH}\\sbert_features_normalized.npy"

# Local model path.
LOCAL_MODEL_PATH = os.environ.get(
    "TIDP_SENTENCE_BERT_MODEL", "all-MiniLM-L6-v2"
)

# ==================== 1. Load data ====================
print("="*60)
print("SBERT feature extraction - single-step operations")
print("="*60)

print(f"\n📂 Loading data: {INPUT_FILE}")
df = pd.read_excel(INPUT_FILE)

# Use the unified_simplified field.
texts = df["unified_simplified"].tolist()
print(f"   Number of samples: {len(texts)}")
print(f"   Unique operation types: {df['unified_simplified'].nunique()}")

# Display text examples.
print(f"\n📝 Text examples (first 10):")
for i in range(min(10, len(texts))):
    print(f"   {i+1}. {texts[i]}")

# ==================== 2. Load the local SBERT model ====================
print("\n" + "="*60)
print("Loading the local SBERT model")
print("="*60)

print(f"Model path: {LOCAL_MODEL_PATH}")

try:
    model = SentenceTransformer(LOCAL_MODEL_PATH, local_files_only=True)
    print("✅ Local model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load the local model: {e}")
    print("   Check whether the model path is correct.")
    exit(1)

# Model information.
print(f"\n📊 Model information:")
print(f"   Maximum sequence length: {model.max_seq_length}")
print(f"   Output dimension: {model.get_sentence_embedding_dimension()}")

# ==================== 3. Extract features ====================
print("\n" + "="*60)
print("Extracting SBERT features")
print("="*60)

print(f"Encoding {len(texts)} texts...")
start_time = time.time()

# Batch encoding.
embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True,
    batch_size=64,
    normalize_embeddings=False
)

end_time = time.time()
print("\n✅ Encoding completed.")
print(f"   Time: {end_time - start_time:.2f} seconds")
print(f"   Feature matrix shape: {embeddings.shape}")
print(f"   Feature dimension: {embeddings.shape[1]}")
print(f"   Feature matrix size: {embeddings.nbytes / 1024 / 1024:.2f} MB")

# ==================== 4. L2 normalization ====================
print("\n" + "="*60)
print("L2 normalization")
print("="*60)

from sklearn.preprocessing import normalize
embeddings_normalized = normalize(embeddings, norm='l2')
print("✅ L2 normalization completed")
print(f"   Normalized feature shape: {embeddings_normalized.shape}")

# ==================== 5. Save features ====================
print("\n" + "="*60)
print("Saving features")
print("="*60)

# Save raw features.
np.save(OUTPUT_FEATURES, embeddings)
print(f"✅ Raw features saved to: {OUTPUT_FEATURES}")

# Save normalized features.
np.save(OUTPUT_FEATURES_NORM, embeddings_normalized)
print(f"✅ Normalized features saved to: {OUTPUT_FEATURES_NORM}")

# ==================== 6. Feature statistics ====================
print("\n" + "="*60)
print("Feature statistics")
print("="*60)

print(f"Feature mean: {embeddings.mean():.6f}")
print(f"Feature standard deviation: {embeddings.std():.6f}")
print(f"Feature minimum: {embeddings.min():.6f}")
print(f"Feature maximum: {embeddings.max():.6f}")

# Check for all-zero vectors.
zero_vectors = np.sum(np.abs(embeddings), axis=1) == 0
if zero_vectors.any():
    print(f"⚠️ Warning: found {zero_vectors.sum()} all-zero vectors")
else:
    print("✅ No all-zero vectors found")

# ==================== 7. Save configuration ====================
config_file = OUTPUT_FEATURES.replace('.npy', '_config.txt')
with open(config_file, 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("SBERT feature extraction configuration\n")
    f.write("="*60 + "\n\n")
    f.write(f"Input file: {INPUT_FILE}\n")
    f.write(f"Model path: {LOCAL_MODEL_PATH}\n")
    f.write(f"Sample count: {len(texts)}\n")
    f.write(f"Unique operation types: {df['unified_simplified'].nunique()}\n")
    f.write(f"Feature dimension: {embeddings.shape[1]}\n\n")
    f.write("Output files:\n")
    f.write(f"  - Raw features: {OUTPUT_FEATURES}\n")
    f.write(f"  - Normalized features: {OUTPUT_FEATURES_NORM}\n\n")
    f.write("Feature statistics:\n")
    f.write(f"  - Mean: {embeddings.mean():.6f}\n")
    f.write(f"  - Standard deviation: {embeddings.std():.6f}\n")

print(f"✅ Configuration saved to: {config_file}")

# ==================== 8. Final report ====================
print("\n" + "="*60)
print("Feature extraction completed.")
print("="*60)

print(f"""
📊 Final summary:
   - Samples: {len(texts)}
   - Unique operation types: {df['unified_simplified'].nunique()}
   - Feature dimension: {embeddings.shape[1]}
   - Feature matrix size: {embeddings.nbytes / 1024 / 1024:.2f} MB
   - Encoding time: {end_time - start_time:.2f} seconds

📁 Output files:
   - Raw features: {OUTPUT_FEATURES}
   - Normalized features: {OUTPUT_FEATURES_NORM}
   - Configuration: {config_file}

Next: run the clustering analysis script.
""")

# ==================== Helper functions ====================
def load_features():
    """Load raw features."""
    features = np.load(OUTPUT_FEATURES)
    print(f"Loaded raw features: {features.shape}")
    return features

def load_normalized_features():
    """Load normalized features."""
    features = np.load(OUTPUT_FEATURES_NORM)
    print(f"Loaded normalized features: {features.shape}")
    return features

if __name__ == "__main__":
    # Optional load test.
    # features = load_features()
    pass
