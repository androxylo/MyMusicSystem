
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from config.settings import load_config
from connectors.tidal import TidalConnector

try:
    cfg = load_config()
    tidal = TidalConnector(cfg)
    playlists = tidal.session.user.playlists()
    discovery = None
    for p in playlists:
        if p.name.lower() == 'mydailydiscovery':
            discovery = p
            break
    
    if discovery:
        print("PLAYLIST_FOUND")
        tracks = discovery.tracks()
        for i, t in enumerate(tracks[:5]):
            print(f"TRACK: {t.name} | ARTIST: {t.artist.name}")
    else:
        print("PLAYLIST_NOT_FOUND")
except Exception as e:
    print("Error:", e)
