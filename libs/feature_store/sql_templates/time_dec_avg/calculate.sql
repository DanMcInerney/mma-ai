-- Time-Decayed Average Calculator SQL template
-- Calculates time-decayed rolling averages for each specified column
-- with exponential decay based on the specified decay rate

WITH base AS (
    SELECT 
        f.fight_id,
        f.fighter_id,
        f.event_id,
        em.event_date,
        {% for col in columns %}
        f.{{ col }}{% if not loop.last %},{% endif %}
        {% endfor %}
    FROM {{ schema }}.{{ table_name }} f
    JOIN {{ schema }}.event_mapping em ON f.event_id = em.event_id
),
current_fights AS (
    SELECT 
        fighter_id,
        fight_id,
        event_id,
        event_date
    FROM base
)
SELECT
    c.fight_id,
    c.fighter_id,
    c.event_id,
    {{ expressions_str }}
FROM current_fights c
LEFT JOIN base b ON c.fighter_id = b.fighter_id
                AND b.event_date <= c.event_date
GROUP BY c.fight_id, c.fighter_id, c.event_id, c.event_date
ORDER BY c.event_date, c.fight_id 