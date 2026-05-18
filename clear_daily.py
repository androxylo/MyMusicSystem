import agent_tools.tools as t

ctx = t._get_ctx()
tidal = ctx.playlist_manager._tidal

playlists = tidal._s.user.playlists()
target = None
for p in playlists:
    print(f"ID: {p.id}, Name: {p.name}")
    if p.name == "MyDailyDiscovery":
        target = p.id

if target:
    print(f"Clearing playlist {target}...")
    tidal.set_playlist_tracks(str(target), [])
    print("Cleared.")
else:
    print("Playlist not found.")
