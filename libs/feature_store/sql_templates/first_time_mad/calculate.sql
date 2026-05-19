CREATE TABLE {{ schema }}.{{ table_name }}{{ table_suffix }} AS
WITH first_fights AS (
    SELECT
        t.fighter_id,
        fm.weightclass,
        e.event_date,
        ROW_NUMBER() OVER (
            PARTITION BY t.fighter_id 
            ORDER BY e.event_date ASC
        ) AS rn,
        t.*
    FROM {{ schema }}.{{ table_name }} t
    JOIN {{ schema }}.fight_mapping fm ON fm.fight_id = t.fight_id
    JOIN {{ schema }}.event_mapping e ON fm.event_id = e.event_id
    WHERE e.event_date BETWEEN :start_date AND :end_date
)
SELECT
    weightclass,
    {{ mad_selects }}
FROM first_fights t
WHERE rn = 1
GROUP BY weightclass; 