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
)
SELECT 
    b.fight_id,
    b.fighter_id,
    b.event_id,
    {{ col_calcs_str }}
FROM base b
LEFT JOIN LATERAL (
    SELECT
        SUM(EXP(-{{ decay_rate }} * ((b.event_date - b2.event_date)::INTEGER / 365.25))) AS sum_w,
        {{ sum_wx_exprs_str }},
        {{ sum_wx2_exprs_str }},
        COUNT(*) AS count_fights
    FROM base b2
    WHERE b2.fighter_id = b.fighter_id
      AND (b2.event_date, b2.fight_id) <= (b.event_date, b.fight_id)
) agg ON TRUE
LEFT JOIN {{ schema }}.{{ table_name }}_first_time_sdev_stats ftss
    ON b.weightclass = ftss.weightclass
ORDER BY b.event_date, b.fight_id 