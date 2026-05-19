-- Template for calculating fighter ape index (reach/height ratio)
-- Parameters:
--   schema: Database schema (e.g., 'features')

WITH fighter_ape AS (
    SELECT 
        fighter_id,
        CAST(
            (CAST(reach AS numeric) / NULLIF(height, 0))::numeric(10,3)
            AS double precision
        ) as ape_index
    FROM {{ schema }}.fight_stats_fe
    WHERE height IS NOT NULL
    AND reach IS NOT NULL
)
UPDATE {{ schema }}.fight_stats_fe f
SET ape = CAST(fa.ape_index AS double precision)
FROM fighter_ape fa
WHERE f.fighter_id = fa.fighter_id 