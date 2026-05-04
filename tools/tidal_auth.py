#!/usr/bin/env python
"""
Run this once to authenticate Tidal.
Usage:  python tools/tidal_auth.py
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tidalapi
except ImportError:
    sys.exit("Run: pip install tidalapi")

token_path = Path(__file__).parent.parent / "data" / "tidal_tokens.json"
token_path.parent.mkdir(exist_ok=True)

session = tidalapi.Session()

# Try loading saved tokens first
if token_path.exists():
    tokens = json.loads(token_path.read_text())
    ok = session.load_oauth_session(
        token_type=tokens["token_type"],
        access_token=tokens["access_token"],
        refresh_token=tokens.get("refresh_token"),
        expiry_time=tokens.get("expiry_time"),
    )
    if ok and session.check_login():
        print("Already authenticated. You're good to go.")
        sys.exit(0)

link_login, future = session.login_oauth()
print("\n" + "="*60)
print("TIDAL LOGIN — open this URL in your browser:")
print()
print(link_login.verification_uri_complete)
print()
print("Waiting for you to log in (Ctrl-C to cancel)...")
print("="*60 + "\n")
sys.stdout.flush()

try:
    future.result()
except KeyboardInterrupt:
    sys.exit("\nCancelled.")

if not session.check_login():
    sys.exit("Authentication failed.")

tokens = {
    "token_type": session.token_type,
    "access_token": session.access_token,
    "refresh_token": session.refresh_token,
    "expiry_time": session.expiry_time.isoformat() if session.expiry_time else None,
}
token_path.write_text(json.dumps(tokens, indent=2))
print(f"Authenticated! Tokens saved to {token_path}")

# Quick smoke test
results = session.search("Aphex Twin", [tidalapi.Track], limit=2)
print("\nSearch test:")
for t in results.get("tracks", []):
    print(f"  {t.artist.name} — {t.name}")
