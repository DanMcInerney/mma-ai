-- Calculate per-minute rates for all count-based statistics
-- Rate is calculated as count / (time_sec/60) and rounded to 4 decimal places

SELECT
    t.fight_id,
    t.fighter_id,
    {% for feat in features %}
    ROUND(CAST(
        t.{{ feat }}::float / 
        (t.{% if '_rd1' in feat %}time_sec_rd1{% else %}time_sec{% endif %}::float/60.0)
    AS NUMERIC), 4) AS {{ feat }}_per_min{% if not loop.last %},
    {% endif %}
    {% endfor %}

FROM {{ schema }}.{{ table_name }} t
JOIN {{ schema }}.event_mapping em ON t.event_id = em.event_id
JOIN {{ schema }}.fight_mapping fm ON t.fight_id = fm.fight_id
ORDER BY em.event_date, t.fight_id
 