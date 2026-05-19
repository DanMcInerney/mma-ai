-- First Time Fighter Average calculation template
-- Calculates averages for each weightclass from first-time fighters' stats
-- from a specified date range.
--
-- Parameters:
-- - {{ schema }}: The database schema (typically 'features')
-- - {{ table_name }}: The base table name (e.g., 'sig_str')
-- - {{ table_suffix }}: The suffix for the output table (typically '_first_time_avg_stats') 
-- - {{ columns }}: The list of columns to calculate averages for
-- - {{ avg_selects }}: Pre-formatted SQL expressions for each column average
-- - {{ start_date }}: Start date for first-time fighter filtering (e.g., '2015-01-01')
-- - {{ end_date }}: End date for first-time fighter filtering (e.g., '2023-01-01')

CREATE TABLE {{ schema }}.{{ table_name }}{{ table_suffix }} AS
WITH first_time_fighters AS (
    SELECT 
        f.fighter_id,
        fm.weightclass,
        {% for col in columns %}
        f.{{ col }}{% if not loop.last %},{% endif %}
        {% endfor %}
    FROM (
        SELECT 
            fighter_id,
            MIN(event_id) as first_event_id
        FROM {{ schema }}.{{ table_name }}
        GROUP BY fighter_id
    ) ff
    JOIN {{ schema }}.{{ table_name }} f
        ON ff.fighter_id = f.fighter_id
        AND ff.first_event_id = f.event_id
    JOIN {{ schema }}.fight_mapping fm
        ON f.fight_id = fm.fight_id
    JOIN {{ schema }}.event_mapping em
        ON f.event_id = em.event_id
    WHERE em.event_date BETWEEN :start_date AND :end_date
)
SELECT 
    weightclass,
    {{ avg_selects }}
FROM first_time_fighters t
GROUP BY weightclass
ORDER BY weightclass; 