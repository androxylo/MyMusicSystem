import sys
import os
sys.path.insert(0, os.path.abspath('.'))

try:
    import tidalapi
    s = tidalapi.Session()
    s.load_oauth_session(*s.check_login()) # this is usually how tidalapi loads if they save the session, but tidal_auth.py says "Already authenticated". Let's check how tidal_auth.py checks login.
except Exception as e:
    print(e)
