-- Template for calculating fighter reach
-- Parameters:
--   schema: Database schema (e.g., 'features')

WITH weightclass_reach AS (
    SELECT 
        fm.weightclass,
        ROUND(AVG(fmap.fighter_reach))::INTEGER as avg_reach
    FROM {{ schema }}.fight_mapping fm
    JOIN {{ schema }}.fighter_mapping fmap ON 
        fm.fighter1_id = fmap.fighter_id OR 
        fm.fighter2_id = fmap.fighter_id
    WHERE fmap.fighter_reach IS NOT NULL
    GROUP BY fm.weightclass
)
UPDATE {{ schema }}.fight_stats_fe f
SET reach = CAST(
    COALESCE(
        m.fighter_reach,
        wr.avg_reach
    ) AS INTEGER
)
FROM {{ schema }}.fighter_mapping m,
     {{ schema }}.fight_mapping fm,
     weightclass_reach wr
WHERE f.fighter_id = m.fighter_id
AND f.fight_id = fm.fight_id
AND wr.weightclass = fm.weightclass 