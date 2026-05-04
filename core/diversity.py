"""
DiversityEnforcer — the sole authority on what makes the final 10.

Design principles:
- Diversity is structural: engines focus on quality, enforcer enforces variety.
- One track per genre_primary bucket per session (hard constraint by default).
- Genres appearing in 3+ recent sessions with avg rating ≥ threshold are
  deprioritized (moved to end of pool), NOT removed. The session can still
  succeed even if every bucket is saturated.
- Engine representation: at least min_engines_represented distinct engines
  must appear in the final selection (relaxed if not enough variety exists).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from core.base_engine import RatedTrack, Session, Suggestion

logger = logging.getLogger(__name__)

# Mapping from fine-grained genre tags (lowercase) → broad genre bucket.
# Used by classify_genre() when genre_primary doesn't directly match a bucket.
GENRE_BUCKETS: dict[str, str] = {
    # Electronic / Dance
    "electronic": "Electronic",
    "electronica": "Electronic",
    "techno": "Electronic",
    "house": "Electronic",
    "trance": "Electronic",
    "ambient": "Electronic",
    "edm": "Electronic",
    "dubstep": "Electronic",
    "drum and bass": "Electronic",
    "dnb": "Electronic",
    "idm": "Electronic",
    "synth": "Electronic",
    "electro": "Electronic",
    "club": "Electronic",
    # Rock
    "rock": "Rock",
    "indie rock": "Rock",
    "alternative rock": "Rock",
    "hard rock": "Rock",
    "punk": "Rock",
    "post-punk": "Rock",
    "emo": "Rock",
    "grunge": "Rock",
    "shoegaze": "Rock",
    "progressive rock": "Rock",
    "classic rock": "Rock",
    "psychedelic rock": "Rock",
    # Metal
    "metal": "Metal",
    "heavy metal": "Metal",
    "death metal": "Metal",
    "black metal": "Metal",
    "doom metal": "Metal",
    "thrash metal": "Metal",
    "metalcore": "Metal",
    "post-metal": "Metal",
    # Pop
    "pop": "Pop",
    "synth-pop": "Pop",
    "dream pop": "Pop",
    "indie pop": "Pop",
    "art pop": "Pop",
    "electropop": "Pop",
    "k-pop": "Pop",
    "j-pop": "Pop",
    # Hip-Hop / R&B
    "hip-hop": "Hip-Hop/R&B",
    "hip hop": "Hip-Hop/R&B",
    "rap": "Hip-Hop/R&B",
    "r&b": "Hip-Hop/R&B",
    "rnb": "Hip-Hop/R&B",
    "soul": "Hip-Hop/R&B",
    "neo soul": "Hip-Hop/R&B",
    "trap": "Hip-Hop/R&B",
    "grime": "Hip-Hop/R&B",
    # Jazz
    "jazz": "Jazz",
    "jazz fusion": "Jazz",
    "bebop": "Jazz",
    "cool jazz": "Jazz",
    "free jazz": "Jazz",
    "nu jazz": "Jazz",
    "smooth jazz": "Jazz",
    # Classical
    "classical": "Classical",
    "contemporary classical": "Classical",
    "baroque": "Classical",
    "opera": "Classical",
    "chamber music": "Classical",
    "orchestral": "Classical",
    "minimalism": "Classical",
    # Folk / Country
    "folk": "Folk/Country",
    "indie folk": "Folk/Country",
    "country": "Folk/Country",
    "americana": "Folk/Country",
    "bluegrass": "Folk/Country",
    "singer-songwriter": "Folk/Country",
    # Blues
    "blues": "Blues",
    "delta blues": "Blues",
    "chicago blues": "Blues",
    "electric blues": "Blues",
    # Reggae / World
    "reggae": "Reggae/World",
    "world": "Reggae/World",
    "world music": "Reggae/World",
    "latin": "Reggae/World",
    "afrobeat": "Reggae/World",
    "ska": "Reggae/World",
    "dub": "Reggae/World",
    "bossa nova": "Reggae/World",
    "samba": "Reggae/World",
    # Experimental / Avant-garde
    "experimental": "Experimental",
    "avant-garde": "Experimental",
    "noise": "Experimental",
    "drone": "Experimental",
    "post-rock": "Experimental",
    "math rock": "Experimental",
    "krautrock": "Experimental",
}


def classify_genre(genre_primary: str, genre_tags: list[str] | None = None) -> str:
    """
    Map a track's genre_primary (and optionally genre_tags) to a broad bucket.

    Falls back to 'Other' if no match is found.
    """
    if genre_primary:
        # Direct match first
        bucket = GENRE_BUCKETS.get(genre_primary.lower())
        if bucket:
            return bucket
        # Check if genre_primary IS already a bucket name
        if genre_primary in set(GENRE_BUCKETS.values()):
            return genre_primary

    # Try genre_tags
    for tag in (genre_tags or []):
        bucket = GENRE_BUCKETS.get(tag.lower())
        if bucket:
            return bucket

    return genre_primary or "Other"


class DiversityEnforcer:
    """
    Selects the final n tracks from a pool of candidates, enforcing:
    - One track per genre_primary bucket (relaxed if pool is too small)
    - Recently-heard tracks excluded (configurable window)
    - Saturated genres deprioritized (moved to end of pool)
    - At least min_engines_represented distinct engines in final selection
    """

    def __init__(
        self,
        recently_heard_days: int = 30,
        saturation_sessions: int = 3,
        saturation_min_avg_rating: float = 7.0,
        min_engines_represented: int = 3,
    ) -> None:
        self._recently_heard_days = recently_heard_days
        self._saturation_sessions = saturation_sessions
        self._saturation_min_avg_rating = saturation_min_avg_rating
        self._min_engines_represented = min_engines_represented

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select(
        self,
        pool: list[Suggestion],
        n: int,
        rated_tracks: list[RatedTrack],
        recent_sessions: list[Session],
        genre_session_history: list[dict],
    ) -> list[Suggestion]:
        """
        Return up to n suggestions from pool, applying all diversity constraints.

        Args:
            pool:                   All candidate suggestions from all engines.
            n:                      Target number of final tracks.
            rated_tracks:           Full rating history (for recently-heard exclusion).
            recent_sessions:        Last N completed sessions.
            genre_session_history:  Output of db.get_genre_session_history().

        Returns:
            list of up to n Suggestion objects with diverse genre coverage.
        """
        if not pool:
            return []

        # Step 1: Remove recently-heard tracks
        recently_heard = self._recently_heard_ids(rated_tracks)
        pool = [s for s in pool if s.track.id not in recently_heard]

        # Step 2: Classify genre buckets
        pool = self._annotate_buckets(pool)

        # Step 3: Identify saturated genres
        saturated = self._saturated_genres(genre_session_history, recent_sessions)

        # Step 4: Split pool into normal and saturated candidates
        normal = [s for s in pool if classify_genre(s.track.genre_primary, s.track.genre_tags) not in saturated]
        deprioritized = [s for s in pool if classify_genre(s.track.genre_primary, s.track.genre_tags) in saturated]

        if saturated:
            logger.debug(f"Saturated genres (deprioritized): {saturated}")

        # Step 5: Greedy one-per-genre selection from normal pool, then deprioritized
        selected = self._greedy_select(normal + deprioritized, n)

        # Step 6: Ensure engine representation (best-effort)
        selected = self._ensure_engine_representation(selected, pool, n)

        return selected

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recently_heard_ids(self, rated_tracks: list[RatedTrack]) -> set[str]:
        cutoff = datetime.now() - timedelta(days=self._recently_heard_days)
        return {
            rt.track.id
            for rt in rated_tracks
            if rt.rated_at is not None and rt.rated_at >= cutoff
        }

    @staticmethod
    def _annotate_buckets(pool: list[Suggestion]) -> list[Suggestion]:
        """Resolve genre_primary for each suggestion using classify_genre."""
        for s in pool:
            s.track.genre_primary = classify_genre(
                s.track.genre_primary, s.track.genre_tags
            )
        return pool

    def _saturated_genres(
        self,
        genre_session_history: list[dict],
        recent_sessions: list[Session],
    ) -> set[str]:
        """
        Return the set of genre bucket names that are 'saturated'.

        A genre is saturated if it appeared in >= saturation_sessions of the
        most recent sessions with avg_rating >= saturation_min_avg_rating.
        """
        if not recent_sessions:
            return set()

        # Only consider the last saturation_sessions sessions
        recent_ids = {s.id for s in recent_sessions[: self._saturation_sessions]}

        # Group history by genre
        genre_session_data: dict[str, list[float]] = {}
        for row in genre_session_history:
            if row["session_id"] not in recent_ids:
                continue
            genre = row["genre"]
            avg = row.get("avg_rating", 0.0) or 0.0
            genre_session_data.setdefault(genre, []).append(avg)

        saturated = set()
        for genre, ratings in genre_session_data.items():
            if (
                len(ratings) >= self._saturation_sessions
                and sum(ratings) / len(ratings) >= self._saturation_min_avg_rating
            ):
                saturated.add(genre)

        return saturated

    def _greedy_select(self, pool: list[Suggestion], n: int) -> list[Suggestion]:
        """
        Greedy selection: iterate pool (ordered by engine_score desc) and pick
        at most one track per genre_primary bucket.

        Falls back to allowing repeats if fewer than n unique buckets exist.
        """
        # Sort descending by engine_score so the best candidates are picked first
        sorted_pool = sorted(pool, key=lambda s: s.engine_score, reverse=True)

        selected: list[Suggestion] = []
        used_genres: set[str] = set()
        leftover: list[Suggestion] = []

        # First pass: one per genre
        seen_track_ids: set[str] = set()
        for suggestion in sorted_pool:
            if suggestion.track.id in seen_track_ids:
                continue
            genre = suggestion.track.genre_primary
            if genre not in used_genres:
                selected.append(suggestion)
                used_genres.add(genre)
                seen_track_ids.add(suggestion.track.id)
                if len(selected) >= n:
                    return selected
            else:
                leftover.append(suggestion)

        # Second pass: relax genre constraint if we still need more
        for suggestion in leftover:
            if suggestion.track.id in seen_track_ids:
                continue
            selected.append(suggestion)
            seen_track_ids.add(suggestion.track.id)
            if len(selected) >= n:
                break

        return selected

    def _ensure_engine_representation(
        self,
        selected: list[Suggestion],
        pool: list[Suggestion],
        n: int,
    ) -> list[Suggestion]:
        """
        Best-effort: if fewer than min_engines_represented engines appear in
        selected, try to swap in tracks from underrepresented engines.
        This is a secondary constraint — it never reduces diversity if the
        swap would reintroduce a duplicate genre.
        """
        if not selected:
            return selected

        represented = {s.engine_name for s in selected}
        if len(represented) >= self._min_engines_represented:
            return selected

        # Find tracks from missing engines in the pool (not already selected)
        selected_ids = {s.track.id for s in selected}
        selected_genres = {s.track.genre_primary for s in selected}
        missing_engines = (
            {s.engine_name for s in pool} - represented
        )

        for candidate in sorted(pool, key=lambda s: s.engine_score, reverse=True):
            if len({s.engine_name for s in selected}) >= self._min_engines_represented:
                break
            if candidate.track.id in selected_ids:
                continue
            if candidate.engine_name not in missing_engines:
                continue
            # Only swap if the genre is not already covered — don't worsen diversity
            if candidate.track.genre_primary in selected_genres:
                continue
            # Replace the lowest-scored track from an over-represented engine
            # (one that has 2+ tracks in selected)
            engine_counts: dict[str, int] = {}
            for s in selected:
                engine_counts[s.engine_name] = engine_counts.get(s.engine_name, 0) + 1
            over_represented = [
                e for e, c in engine_counts.items() if c > 1
            ]
            if not over_represented:
                # Can't swap without reducing representation
                break
            # Find the weakest track from an over-represented engine
            victims = [
                s for s in selected if s.engine_name in over_represented
            ]
            victim = min(victims, key=lambda s: s.engine_score)
            selected = [s for s in selected if s is not victim]
            selected.append(candidate)
            selected_ids.discard(victim.track.id)
            selected_ids.add(candidate.track.id)
            selected_genres.discard(victim.track.genre_primary)
            selected_genres.add(candidate.track.genre_primary)
            missing_engines.discard(candidate.engine_name)

        return selected
