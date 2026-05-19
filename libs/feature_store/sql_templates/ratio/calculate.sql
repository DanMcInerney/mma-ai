-- Ratio Calculator SQL Template
-- Calculates fighter_stat / (fighter_stat + opponent_stat)
-- Parameters:
--   schema: Database schema (e.g., 'features')
--   table_name: Table name (e.g., 'fight_stats_derived')
--   features: List of features to calculate ratios for
--   exclude_patterns: List of patterns to exclude from opponent stats

WITH opponent_stats AS (
    -- Get opponent stats using feature_utils.get_opponent_stats
    {% if feature_utils %}
    {{ feature_utils.get_opponent_stats(table_name, exclude_patterns) }}
    {% else %}
    -- Fallback when feature_utils is not available
    WITH fighter_data AS (
        SELECT
            f.fight_id,
            f.fighter_id,
            fm.fighter1_id,
            fm.fighter2_id,
            CASE
                WHEN f.fighter_id = fm.fighter1_id THEN fm.fighter2_id
                WHEN f.fighter_id = fm.fighter2_id THEN fm.fighter1_id
            END AS opponent_id
        FROM {{ schema }}.{{ table_name }} f
        JOIN {{ schema }}.fight_mapping fm ON f.fight_id = fm.fight_id
    )
    SELECT
        fd.fight_id,
        fd.fighter_id,
        {% for col in features %}
        opp.{{ col }}{% if not loop.last %},{% endif %}
        {% endfor %}
    FROM fighter_data fd
    JOIN {{ schema }}.{{ table_name }} opp 
        ON fd.fight_id = opp.fight_id 
        AND fd.opponent_id = opp.fighter_id
    {% endif %}
)
SELECT 
    t.fight_id,
    t.fighter_id,
    {% for col in features %}
    CAST(
        CASE WHEN (t.{{ col }} + opp.{{ col }}) > 0 
            THEN ROUND((t.{{ col }}::numeric / (t.{{ col }} + opp.{{ col }})::numeric), 3)
            ELSE 0 
        END
    AS FLOAT) AS {{ col }}_ratio{% if not loop.last %},
    {% endif %}
    {% endfor %}

FROM {{ schema }}.{{ table_name }} t
JOIN opponent_stats opp 
    ON t.fight_id = opp.fight_id 
    AND t.fighter_id = opp.fighter_id
JOIN {{ schema }}.fight_mapping fm ON t.fight_id = fm.fight_id
JOIN {{ schema }}.event_mapping em ON fm.event_id = em.event_id
ORDER BY em.event_date, t.fight_id 