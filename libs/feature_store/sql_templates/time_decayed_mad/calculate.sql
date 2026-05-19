WITH base AS (
    SELECT 
        f.fight_id,
        f.fighter_id,
        f.event_id,
        em.event_date,
        fm.weightclass,
        {{ features_str }}
    FROM {{ schema }}.{{ table_name }} f {{ sampling }}
    JOIN {{ schema }}.event_mapping em ON f.event_id = em.event_id
    JOIN {{ schema }}.fight_mapping fm ON f.fight_id = fm.fight_id
),

{{ weighted_mad_calcs_str }}

SELECT 
    b.fight_id,
    b.fighter_id,
    b.event_id,
    {{ col_calcs_str }}
FROM base b
{% for col in calc_columns %}
LEFT JOIN mad_{{ col }} m_{{ col }} ON b.fight_id = m_{{ col }}.fight_id AND b.fighter_id = m_{{ col }}.fighter_id
{% endfor %}
LEFT JOIN {{ schema }}.{{ table_name }}_first_time_mad_stats ftms
    ON b.weightclass = ftms.weightclass
ORDER BY b.event_date, b.fight_id 