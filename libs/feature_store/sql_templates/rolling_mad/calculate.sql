WITH fight_data AS (
    SELECT
        f.fight_id,
        f.fighter_id,
        f.event_id,
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
all_fights AS (
    SELECT 
        curr.fight_id,
        curr.fighter_id,
        curr.event_id,
        curr.event_date,
        curr.weightclass,
        curr.rn,
        {% for col in columns %}
        curr.{{ col }},
        {% endfor %}
        prev.fight_id as prev_fight_id,
        prev.fighter_id as prev_fighter_id,
        {% for col in columns %}
        prev.{{ col }} as prev_{{ col }}{% if not loop.last %},{% endif %}
        {% endfor %}
    FROM fight_data curr
    LEFT JOIN fight_data prev 
        ON curr.fighter_id = prev.fighter_id 
        AND (prev.event_date < curr.event_date OR (prev.event_date = curr.event_date AND prev.fight_id <= curr.fight_id))
),
fighter_medians AS (
    SELECT
        curr.fight_id,
        curr.fighter_id,
        curr.event_date,
        curr.weightclass,
        curr.rn,
        {% for col in columns %}
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY prev.{{ col }}) AS {{ col }}_median{% if not loop.last %},{% endif %}
        {% endfor %}
    FROM fight_data curr
    JOIN fight_data prev 
        ON curr.fighter_id = prev.fighter_id 
        AND (prev.event_date < curr.event_date OR (prev.event_date = curr.event_date AND prev.fight_id <= curr.fight_id))
    GROUP BY curr.fight_id, curr.fighter_id, curr.event_date, curr.weightclass, curr.rn
),
fighter_mads AS (
    SELECT
        f.fight_id,
        f.fighter_id,
        f.event_date,
        f.weightclass,
        f.rn,
        {% for col in columns %}
        CASE 
            WHEN f.rn = 1 THEN COALESCE(ftms.{{ col }}_wc_mad, 0) 
            ELSE PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ABS(a.{{ col }} - m.{{ col }}_median)) 
        END AS {{ col }}_mad{% if not loop.last %},{% endif %}
        {% endfor %}
    FROM fighter_medians m
    JOIN fight_data f ON f.fight_id = m.fight_id AND f.fighter_id = m.fighter_id
    JOIN all_fights a ON a.fighter_id = f.fighter_id AND (a.event_date < f.event_date OR (a.event_date = f.event_date AND a.fight_id <= f.fight_id))
    LEFT JOIN {{ schema }}.{{ table_name }}_first_time_mad_stats ftms ON f.weightclass = ftms.weightclass
    GROUP BY f.fight_id, f.fighter_id, f.event_date, f.weightclass, f.rn, 
        {% for col in columns %}
        ftms.{{ col }}_wc_mad{% if not loop.last %},{% endif %}
        {% endfor %}
)
SELECT
    fight_id,
    fighter_id,
    {% for col in columns %}
    {{ col }}_mad{% if not loop.last %},{% endif %}
    {% endfor %}
FROM fighter_mads
ORDER BY event_date, fight_id 