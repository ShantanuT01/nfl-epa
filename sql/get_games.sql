SELECT 
    game_id,
    season,
    game_type,
    week,
    gameday,
    CASE 
        WHEN away_team = 'SD' THEN 'LAC'
        WHEN away_team = 'STL' THEN 'LA'
        WHEN away_team = 'OAK' THEN 'LV'
        ELSE away_team 
        END
        AS away_team,
    CASE 
        WHEN home_team = 'SD' THEN 'LAC'
        WHEN home_team = 'STL' THEN 'LA'
        WHEN home_team = 'OAK' THEN 'LV'
        ELSE home_team 
        END
        AS home_team,
    home_score,
    away_score,
    away_rest,
    home_rest,
    away_moneyline,
    home_moneyline,
    spread_line,
    away_spread_odds,
    home_spread_odds,
    total_line,
    under_odds,
    over_odds,
    div_game
FROM 'games.parquet'
WHERE season >= 2002 AND home_score IS NOT NULL and away_score IS NOT NULL
ORDER BY season, gameday

