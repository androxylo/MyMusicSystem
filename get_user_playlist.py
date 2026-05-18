import json, sys
from pathlib import Path
import tidalapi

token_path = Path("data/tidal_tokens.json")
session = tidalapi.Session()
if token_path.exists():
    tokens = json.loads(token_path.read_text())
    session.load_oauth_session(
        token_type=tokens["token_type"],
        access_token=tokens["access_token"],
        refresh_token=tokens.get("refresh_token"),
        expiry_time=tokens.get("expiry_time"),
    )

if not session.check_login():
    print("Not logged in.")
    sys.exit(1)

playlist_id = "662bdb96-cca6-4460-b991-9d06d12d92cd"
try:
    pl = session.playlist(playlist_id)
    tracks = pl.tracks()
    print(f"Tracks found: {len(tracks)}")
    for i, t in enumerate(tracks):
        print(f"{i+1}. {t.artist.name} - {t.name}")
except Exception as e:
    print(f"Error fetching playlist: {e}")
