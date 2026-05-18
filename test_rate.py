
import agent_tools.tools as t
t.set_curated_threshold(7)
ctx = t._get_ctx()
tracks = ctx.db.conn.execute("SELECT id, title, artist, genre_primary FROM tracks WHERE artist LIKE '%Nick Cave%'").fetchall()
for tr in tracks:
    sugg = ctx.db.conn.execute("SELECT session_id FROM engine_suggestions WHERE track_id = ?", (tr[0],)).fetchone()
    if sugg:
        print(f"Found track: {tr[1]} - {tr[3]}")
        try:
            res = t.rate_track(sugg[0], tr[0], 7)
            print('Rate result:', res)
        except Exception as e:
            print('Rate track failed:', e)

