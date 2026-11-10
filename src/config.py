from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "experimentiq.db"
SEED = 20261022
DATA_DIR.mkdir(exist_ok=True)
