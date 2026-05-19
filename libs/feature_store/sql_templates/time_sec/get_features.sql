SELECT 
    fm.fight_id,
    fs.fighter_id,
    fm.end_round,
    fm.end_time,
    fm.time_format
FROM {{ schema }}.fight_mapping fm
JOIN {{ schema }}.fight_stats_core fs ON fm.fight_id = fs.fight_id 