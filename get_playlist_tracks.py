import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import agent_tools.tools as t

try:
    ctx = t._get_ctx()
    playlist_id = "662bdb96-cca6-4460-b991-9d06d12d92cd"
    import tidalapi
    pl = tidalapi.Playlist(ctx.tidal._session, playlist_id)
    tracks = pl.tracks()
    
    print(f"Tracks found: {len(tracks)}")
    for i, tr in enumerate(tracks):
        print(f"{i+1}. {tr.artist.name} - {tr.name}")
except Exception as e:
    print(f"Error: {e}")
