-- Average Calculator SQL template
-- Calculates rolling career averages for each specified column

SELECT 
    f.fight_id,
    f.fighter_id,
    f.event_id,
    {% for calc in column_calcs %}
    {{ calc }}{% if not loop.last %},{% endif %}
    {% endfor %}
FROM {{ schema }}.{{ table_name }} f
JOIN {{ schema }}.event_mapping em ON f.event_id = em.event_id
ORDER BY em.event_date, fight_id 