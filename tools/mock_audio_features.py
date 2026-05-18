import sqlite3
import json

db = sqlite3.connect('./data/music.db')
c = db.cursor()

mock_features = json.dumps({"features": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]})

c.execute('UPDATE tracks SET audio_features = ?', (mock_features,))
db.commit()

print(f"Updated {c.rowcount} tracks with mock audio features.")
db.close()
