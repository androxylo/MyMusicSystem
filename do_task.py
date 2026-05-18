import agent_tools.tools as t

ctx = t._get_ctx()
tidal = ctx.playlist_manager._tidal

# Clear MyDailyDiscovery
daily_id = "662bdb96-cca6-4460-b991-9d06d12d92cd"
try:
    tidal.set_playlist_tracks(daily_id, [])
    print("Cleared MyDailyDiscovery playlist.")
except Exception as e:
    print(f"Error clearing playlist: {e}")

# Find a track and put into Now Rating
# We will use lastfm from the context to find a track similar to something they like, e.g., Max Richter
try:
    now_rating_id = "5447a6ba-e25e-460a-b22f-46a5b4eec447"
    
    # Just grab a track directly from Tidal for simplicity if we can search
    # or let's search Tidal for a track like "The National - Fake Empire" or similar
    search_results = tidal._s.search("Fake Empire", models=[tidal._s.SEARCH_TRACK])
    if search_results and search_results['tracks']:
        track = search_results['tracks'][0]
        print(f"Found track: {track.name} by {track.artist.name}")
        tidal.set_playlist_tracks(now_rating_id, [str(track.id)])
        print("Added track to Now Rating playlist.")
    else:
        print("Could not find track.")
except Exception as e:
    print(f"Error finding/adding track: {e}")
