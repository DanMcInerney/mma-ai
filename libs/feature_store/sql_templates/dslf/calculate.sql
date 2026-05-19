-- Template for calculating days since last fight
-- Parameters:
--   schema: Database schema (e.g., 'features')

WITH ordered_fights AS (
    SELECT 
        f.fighter_id,
        f.fight_id,
        f.event_id,
        e.event_date,
        LAG(e.event_date) OVER (
            PARTITION BY f.fighter_id 
            ORDER BY e.event_date ASC, f.fight_id ASC
        ) as prev_fight_date,
        ROW_NUMBER() OVER (
            PARTITION BY f.fighter_id 
            ORDER BY e.event_date ASC, f.fight_id ASC
        ) as fight_order
    FROM {{ schema }}.fight_stats_fe f
    JOIN {{ schema }}.event_mapping e ON e.event_id = f.event_id
)
UPDATE {{ schema }}.fight_stats_fe f
SET days_since_last_fight = 
    CASE 
        WHEN of.fight_order = 1 THEN 120  -- first fight
        ELSE CAST((of.event_date - of.prev_fight_date) AS INTEGER)
    END
FROM ordered_fights of
WHERE f.fight_id = of.fight_id
AND f.fighter_id = of.fighter_id 