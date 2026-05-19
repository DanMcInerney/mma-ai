-- Template for calculating fighter age at the time of each fight
-- Parameters:
--   schema: Database schema (e.g., 'features')

WITH age_calc AS (
    SELECT 
        f.fight_id,
        f.fighter_id,
        fm.weightclass,
        ROUND(
            EXTRACT(EPOCH FROM AGE(
                e.event_date::timestamp, 
                m.fighter_dob::timestamp
            )) / (365.25 * 24 * 60 * 60)::DECIMAL
            , 3
        ) as calculated_age
    FROM {{ schema }}.fight_stats_fe f
    JOIN {{ schema }}.fighter_mapping m ON f.fighter_id = m.fighter_id
    JOIN {{ schema }}.event_mapping e ON f.event_id = e.event_id
    JOIN {{ schema }}.fight_mapping fm ON f.fight_id = fm.fight_id
),
weight_class_avgs AS (
    SELECT 
        weightclass,
        AVG(calculated_age) as avg_age
    FROM age_calc 
    WHERE calculated_age IS NOT NULL
    GROUP BY weightclass
)
UPDATE {{ schema }}.fight_stats_fe f
SET age = COALESCE(
    (SELECT calculated_age 
     FROM age_calc ac 
     WHERE ac.fight_id = f.fight_id 
     AND ac.fighter_id = f.fighter_id),
    (SELECT avg_age 
     FROM weight_class_avgs wca 
     WHERE wca.weightclass = (
        SELECT fm.weightclass 
        FROM {{ schema }}.fight_mapping fm 
        WHERE fm.fight_id = f.fight_id
     ))
) 