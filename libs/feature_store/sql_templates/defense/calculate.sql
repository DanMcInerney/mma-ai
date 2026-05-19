-- Defense Calculator SQL Template
-- Gets opponent's accuracy as the fighter's defense metric
-- Parameters:
--   schema: Database schema (e.g., 'features')
--   table_name: Table name (e.g., 'fight_stats_derived')
--   features: List of accuracy features to convert to defense metrics

SELECT 
    t.fight_id,
    t.fighter_id,
    {% for feat in features %}
    opp.{{ feat }} AS {{ feat.replace('_acc', '_def') }}{% if not loop.last %},
    {% endif %}
    {% endfor %}

FROM {{ schema }}.{{ table_name }} t
JOIN {{ schema }}.{{ table_name }} opp 
    ON t.fight_id = opp.fight_id 
    AND t.fighter_id != opp.fighter_id
JOIN {{ schema }}.event_mapping em ON t.event_id = em.event_id
ORDER BY em.event_date, t.fight_id 