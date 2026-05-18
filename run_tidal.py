import os
import sys

sys.path.append(os.path.expanduser('~/MyMusicSystem'))
from connectors.tidal import TidalConnector

try:
    tidal = TidalConnector({})
    tidal.authenticate()
    print("Authenticated successfully.")

    daily_discovery_id = "662bdb96-cca6-4460-b991-9d06d12d92cd"
    tidal.set_playlist_tracks(daily_discovery_id, [])
    print("Cleared MyDailyDiscovery")

    search_results = tidal.search_tracks("Angel Massive Attack", limit=5)
    if search_results:
        best_track = search_results[0]
        print(f"Found track: {best_track.title} by {best_track.artist} (ID: {best_track.tidal_id})")
        now_rating_id = "5447a6ba-e25e-460a-b22f-46a5b4eec447"
        tidal.add_tracks_to_playlist(now_rating_id, [best_track.tidal_id])
        print(f"Added to Now Rating: {now_rating_id}")
    else:
        print("Could not find the track.")
except Exception as e:
    print(f"Error: {e}")
