SELECT 
    fm.fight_id,
    fs.fighter_id,
    fm.method,
    fm.result,
    fm.fighter1_id,
    fm.fighter2_id,
    fm.end_round
FROM {{ schema }}.fight_mapping fm
JOIN {{ schema }}.fight_stats_core fs ON fm.fight_id = fs.fight_id 
