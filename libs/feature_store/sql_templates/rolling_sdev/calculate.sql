WITH fight_data AS (
    SELECT
        f.fight_id,
        f.fighter_id,
        em.event_date,
        fm.weightclass,
        ROW_NUMBER() OVER (
            PARTITION BY f.fighter_id
            ORDER BY em.event_date ASC, f.fight_id ASC
        ) AS rn,
        {{ features_str }}
    FROM {{ schema }}.{{ table_name }} f
    JOIN {{ schema }}.event_mapping em ON f.event_id = em.event_id
    JOIN {{ schema }}.fight_mapping fm ON f.fight_id = fm.fight_id
),
joined_fighter_std AS (
    SELECT
        fd.*,
        {{ stddev_exprs_str }}
    FROM fight_data fd
    LEFT JOIN {{ schema }}.{{ table_name }}_first_time_sdev_stats ftss
        ON fd.weightclass = ftss.weightclass
)
SELECT
    fight_id,
    fighter_id,
    {{ output_columns }}
FROM joined_fighter_std
ORDER BY event_date, fight_id 