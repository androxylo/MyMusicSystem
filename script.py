import sqlite3
import json

db_path = 'data/music.db'

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check ratings schema
    cursor.execute('PRAGMA table_info(ratings);')
    schema = cursor.fetchall()
    print('Ratings Schema:', schema)
    
    conn.close()
except Exception as e:
    print('Error:', e)

