import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # classifiers/
sys.path.insert(0, str(ROOT / "mlp_baseline" / "src"))
sys.path.insert(0, str(ROOT / "knn_baseline"))
