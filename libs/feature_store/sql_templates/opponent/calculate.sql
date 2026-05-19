-- Opponent Calculator SQL Template
-- Gets opponent's stats for each fighter in feature-specific tables
-- Parameters:
--   schema: Database schema (e.g., 'features')
--   table_name: Source feature-specific table name
--   columns: List of columns to get opponent values for
--   column_selects: Formatted column selects to prevent SQL syntax errors
--   exclude_patterns: List of patterns to exclude

WITH base_stats AS (
    SELECT 
        t.fight_id,
        t.fighter_id,
        t.event_id
    FROM {{ schema }}.{{ table_name }} t
),
opponent_stats AS (
    SELECT 
        t.fight_id,
        t.fighter_id,
        {{ column_selects }}
    FROM {{ schema }}.{{ table_name }} t
    JOIN {{ schema }}.fight_mapping fm ON t.fight_id = fm.fight_id
    JOIN {{ schema }}.{{ table_name }} opp 
        ON t.fight_id = opp.fight_id 
        AND (
            (t.fighter_id = fm.fighter1_id AND opp.fighter_id = fm.fighter2_id) OR
            (t.fighter_id = fm.fighter2_id AND opp.fighter_id = fm.fighter1_id)
        )
)
SELECT 
    t.fight_id,
    t.fighter_id,
{% for col in columns %}
    opp.{{ col }}_opp{% if not loop.last %},{% endif %}
{% endfor %}

FROM base_stats t
JOIN opponent_stats opp 
    ON t.fight_id = opp.fight_id 
    AND t.fighter_id = opp.fighter_id
JOIN {{ schema }}.event_mapping em ON t.event_id = em.event_id
ORDER BY em.event_date, t.fight_id 