import sys
from pathlib import Path
sys.path.insert(0, str(Path("~/MyMusicSystem").expanduser()))
from agent_tools.tools import start_session
try:
    res = start_session(diversity_mode="relaxed", notes="Adding more tracks")
    print("Success:", res["suggestions"][:3])
except Exception as e:
    print("Error:", e)
