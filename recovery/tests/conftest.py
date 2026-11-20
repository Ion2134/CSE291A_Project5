import sys
from pathlib import Path

# Running from recovery/ puts it on sys.path, where the state_db/ subfolder would shadow the real
# state_db module (as an empty namespace package). Put the module's own folder first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "state_db"))
