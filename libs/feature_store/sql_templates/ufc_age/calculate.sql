-- Template for calculating fighter UFC age (time since UFC debut)
-- Parameters:
--   schema: Database schema (e.g., 'features')

WITH first_fights AS (
    -- For each fighter, determine the debut (first UFC fight date)
    SELECT 
        f.fighter_id,
        MIN(e.event_date) AS first_ufc_fight_date
    FROM {{ schema }}.fight_stats_fe f
    JOIN {{ schema }}.event_mapping e 
      ON f.event_id = e.event_id
    GROUP BY f.fighter_id
),
ufc_age_calc AS (
    -- For each fight, calculate the time in years since the fighter's debut.
    SELECT 
        f.fight_id,
        f.fighter_id,
        ROUND(
            EXTRACT(EPOCH FROM (e.event_date::timestamp - ff.first_ufc_fight_date::timestamp))
            / (365.25 * 24 * 60 * 60)::DECIMAL,
            3
        ) AS calculated_ufc_age
    FROM {{ schema }}.fight_stats_fe f
    JOIN {{ schema }}.event_mapping e 
      ON f.event_id = e.event_id
    JOIN first_fights ff 
      ON f.fighter_id = ff.fighter_id
)
UPDATE {{ schema }}.fight_stats_fe f
SET ufcage = (
    SELECT calculated_ufc_age 
    FROM ufc_age_calc ua 
    WHERE ua.fight_id = f.fight_id 
      AND ua.fighter_id = f.fighter_id
) 