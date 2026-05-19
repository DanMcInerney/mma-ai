-- Template for calculating fighter height
-- Parameters:
--   schema: Database schema (e.g., 'features')

WITH weightclass_height AS (
    SELECT 
        fm.weightclass,
        ROUND(AVG(fmap.fighter_height))::INTEGER as avg_height
    FROM {{ schema }}.fight_mapping fm
    JOIN {{ schema }}.fighter_mapping fmap ON 
        fm.fighter1_id = fmap.fighter_id OR 
        fm.fighter2_id = fmap.fighter_id
    WHERE fmap.fighter_height IS NOT NULL
    GROUP BY fm.weightclass
)
UPDATE {{ schema }}.fight_stats_fe f
SET height = CAST(
    COALESCE(
        m.fighter_height,
        wr.avg_height
    ) AS INTEGER
)
FROM {{ schema }}.fighter_mapping m,
     {{ schema }}.fight_mapping fm,
     weightclass_height wr
WHERE f.fighter_id = m.fighter_id
AND f.fight_id = fm.fight_id
AND wr.weightclass = fm.weightclass 