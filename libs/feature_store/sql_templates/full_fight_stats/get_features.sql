-- Template for retrieving all columns from the fight_stats_fe table
-- This is used by the FullFightStatsCalculator to get all columns including round-specific ones
-- Parameters:
--   schema: Database schema (e.g., 'features')
--   table: Table name (e.g., 'fight_stats_fe')

SELECT *
FROM {{ schema }}.{{ table }} 