"""video-ranker: a two-stage retrieval -> ranking recommender on MovieLens."""
import os

# Defensive: if a second OpenMP runtime is ever loaded alongside PyTorch's
# (LightGBM / implicit on macOS), allow it rather than aborting. The clean fix
# is scripts/fix_macos_openmp.py, which makes a single libomp image load.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
