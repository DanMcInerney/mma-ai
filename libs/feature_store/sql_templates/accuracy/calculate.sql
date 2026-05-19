-- Calculate accuracy percentages for all landed vs. attempted stats
-- Accuracy is calculated as (landed / attempted) * 100 and rounded to 4 decimal place

SELECT
    {{ table_name }}.fight_id,
    {{ table_name }}.fighter_id,
    {% for land_col, att_col in land_att_pairs %}
    ROUND(CAST(
        CASE 
            WHEN {{ table_name }}.{{ att_col }} > 0 THEN
                ({{ table_name }}.{{ land_col }}::float * 100.0 / {{ table_name }}.{{ att_col }}::float)
            ELSE
                0
        END
    AS NUMERIC), 4) AS {% if '_rd1' in land_col %}{{ land_col.replace('_land_rd1', '_rd1_acc') }}{% else %}{{ land_col.replace('_land', '_acc') }}{% endif %}{% if not loop.last %},
    {% endif %}
    {% endfor %}

FROM {{ schema }}.{{ table_name }}
JOIN {{ schema }}.event_mapping ON {{ table_name }}.event_id = event_mapping.event_id
JOIN {{ schema }}.fight_mapping ON {{ table_name }}.fight_id = fight_mapping.fight_id
ORDER BY event_mapping.event_date, {{ table_name }}.fight_id 