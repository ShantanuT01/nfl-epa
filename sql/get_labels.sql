WITH
  drives AS (
    SELECT
      game_id,
      game_half,
      drive,
      posteam,
      defteam,
      fixed_drive_result
    FROM
      (
        SELECT
          game_id,
          game_half,
          drive,
          posteam,
          defteam,
          fixed_drive_result,
          ROW_NUMBER() OVER (
            PARTITION BY
              game_id,
              game_half,
              drive
            ORDER BY
              play_id ASC
          ) AS rn
        FROM
          'pbp.parquet'
        WHERE
          season >= 2016
          AND posteam IS NOT NULL
          AND defteam IS NOT NULL
          AND drive IS NOT NULL
          AND game_id IS NOT NULL
          AND fixed_drive_result IS NOT NULL
      )
    WHERE
      rn = 1
  ),
  scored AS (
    SELECT
      *,
      CASE
        WHEN fixed_drive_result IN ('Touchdown', 'Field goal', 'Safety', 'Opp touchdown', 'End of half') THEN fixed_drive_result
      END AS scoring_result,
      CASE
        WHEN fixed_drive_result IN ('Touchdown', 'Field goal') THEN posteam
        WHEN fixed_drive_result IN ('Opp touchdown', 'Safety') THEN defteam
        WHEN fixed_drive_result = 'End of half' THEN NULL
        ELSE NULL
      END AS scoring_team
    FROM
      drives
  ),
  next_scores AS (
    SELECT
      game_id,
      game_half,
      drive,
      posteam,
      defteam,
      fixed_drive_result,
      FIRST_VALUE (scoring_result IGNORE NULLS) OVER (
        PARTITION BY
          game_id,
          game_half
        ORDER BY
          drive ASC ROWS BETWEEN CURRENT ROW
          AND UNBOUNDED FOLLOWING
      ) AS next_score,
      FIRST_VALUE (scoring_team IGNORE NULLS) OVER (
        PARTITION BY
          game_id,
          game_half
        ORDER BY
          drive ASC ROWS BETWEEN CURRENT ROW
          AND UNBOUNDED FOLLOWING
      ) AS next_scoring_team
    FROM
      scored
    ORDER BY
      game_id,
      drive
  ),
  ep_drive_labels AS (
    SELECT
      *,
      CASE
        WHEN next_scoring_team = posteam
        AND next_score = 'Field goal' THEN 5
        WHEN next_scoring_team = defteam
        AND next_score = 'Field goal' THEN 1
        WHEN next_scoring_team = defteam
        AND next_score = 'Touchdown' THEN 0
        WHEN next_scoring_team = posteam
        AND next_score = 'Touchdown' THEN 6
        WHEN next_scoring_team = defteam
        AND next_score = 'Opp touchdown' THEN 0
        WHEN next_scoring_team = posteam
        AND next_score = 'Opp touchdown' THEN 6
        WHEN next_scoring_team = defteam
        AND next_score = 'Safety' THEN 2
        WHEN next_scoring_team = posteam
        AND next_score = 'Safety' THEN 4
        ELSE 3
      END AS next_score_label
    FROM
      next_scores
  )
SELECT
  pbp.game_id,
  pbp.season, 
  pbp.play_id,
  pbp.drive,
  pbp.fixed_drive_result,
  pbp.home_team,
  pbp.away_team,
  pbp.game_half,
  pbp.week,
  pbp.play_type,
  pbp.season_type,
  pbp.yardline_100,
  pbp.yrdln,
  pbp.down,
  pbp.goal_to_go,
  pbp.time,
  pbp.ydstogo,
  pbp.posteam,
  pbp.posteam_type,
  pbp.defteam,
  pbp.sp,
  pbp.qtr, 
  pbp.field_goal_result,
  pbp.extra_point_attempt,
  pbp.extra_point_result,
  pbp.two_point_attempt,
  pbp.two_point_conv_result,
  pbp.posteam_score,
  pbp.defteam_score,
  pbp.posteam_score - pbp.defteam_score AS score_diff, 
  pbp.posteam_score_post,
  pbp.defteam_score_post,
  pbp.no_score_prob,
  pbp.opp_fg_prob,
  pbp.opp_safety_prob,
  pbp.opp_td_prob,
  pbp.fg_prob,
  pbp.safety_prob,
  pbp.td_prob,
  ep.next_score,
  ep.next_scoring_team,
  ep.next_score_label,
  CASE WHEN pbp.home_team = pbp.posteam THEN 1 ELSE 0 END AS home_advantage,
   -- Split time string into minutes and seconds
  CAST(SPLIT_PART(time, ':', 1) AS INTEGER) AS minutes,
  CAST(SPLIT_PART(time, ':', 2) AS INTEGER) AS seconds,
  
  -- Seconds left in half
  CASE
    WHEN qtr IN (1, 3) THEN
      -- First quarter of each half: full quarter remaining + time on clock
      (CAST(SPLIT_PART(time, ':', 1) AS INTEGER) * 60 +
       CAST(SPLIT_PART(time, ':', 2) AS INTEGER)) + 900
    WHEN qtr IN (2, 4) THEN
      -- Second quarter of each half: just time on clock
      (CAST(SPLIT_PART(time, ':', 1) AS INTEGER) * 60 +
       CAST(SPLIT_PART(time, ':', 2) AS INTEGER))
    WHEN qtr = 5 THEN
      -- Overtime
      (CAST(SPLIT_PART(time, ':', 1) AS INTEGER) * 60 +
       CAST(SPLIT_PART(time, ':', 2) AS INTEGER))
  END AS seconds_left_in_half
FROM
  'pbp.parquet' pbp
INNER JOIN ep_drive_labels ep ON pbp.game_id = ep.game_id
  AND ep.game_half = pbp.game_half
  AND pbp.drive = ep.drive
WHERE
  pbp.season >= 2016
  AND pbp.play_type != 'kickoff'
  AND pbp.play_type != 'no_play'
  
  