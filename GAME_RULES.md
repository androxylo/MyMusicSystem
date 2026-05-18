# Music Rating Game Rules and Taste Profile

## Core Workflows
1. **Standard Game:** Suggest track via Last.fm, queue to "Now Rating", guess rating.
2. **Daily Discovery Game:** Clean up yesterday's tracks in "MyDailyDiscovery", read today's, research genre/metadata, guess ratings.

## Mechanics
- **Threshold:** 7 or higher -> Add to `Liked — <genre>` (em-dash).
- **Weighting:** Exponential curve. 8+ ratings heavily steer recommendations.

## Taste Profile
- **Loves (8+):** Moody, atmospheric, tension, raw vocals, dark indie, post-punk, modern classical, trip-hop, organic electronic.
- **Appreciates (7):** Gritty blues rock, acoustic folk, dynamic tension.
- **Dislikes (<7):** Over-polished mainstream pop, commercial R&B, generic pop-country.
