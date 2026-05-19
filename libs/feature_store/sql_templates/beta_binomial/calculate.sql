-- Beta-Binomial smoothing calculator template
-- Applies Beta-Binomial smoothing to binary outcome stats with weight class specific priors

WITH
{{ prior_cte }}

SELECT
    fd.fight_id,
    fd.fighter_id,
    {% for expr in smoothing_expressions %}
    {{ expr }}{% if not loop.last %},
    {% endif %}
    {% endfor %}
FROM {{ schema }}.{{ table_name }} fd
JOIN {{ schema }}.fight_mapping fm ON fd.fight_id = fm.fight_id
LEFT JOIN weightclass_priors wp ON fm.weightclass = wp.weightclass
CROSS JOIN global_priors gp
ORDER BY fd.fight_id, fd.fighter_id
