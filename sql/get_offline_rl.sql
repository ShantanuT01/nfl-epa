WITH current_state AS (
  SELECT
    season, game_id, season_type, game_date, play_id, drive, next_scoring_drive,
    play_type, down, ydstogo, yardline_100,
    score_diff,posteam_timeouts_remaining, defteam_timeouts_remaining,
    posteam_score_post - defteam_score_post AS score_diff_post,
    score_diff_post - score_diff AS reward,
    posteam, defteam, seconds_left_in_half, timeout, game_half,
    ROW_NUMBER() OVER (
      PARTITION BY season, game_id
      ORDER BY drive, play_id
    ) AS rn
  FROM 'data/ep_labels.parquet'
  WHERE play_type NOT IN ('extra_point', 'qb_kneel', 'qb_spike')
    AND COLUMNS(c -> c IN ('down', 'ydstogo', 'yardline_100','score_diff','seconds_left_in_half','game_half','play_type','reward')) IS NOT NULL
),

joined AS (
  SELECT
    cur.*,
    nxt.play_id          AS next_play_id,
    nxt.drive            AS next_drive,
    nxt.down             AS next_down,
    nxt.ydstogo          AS next_ydstogo,
    nxt.yardline_100     AS next_yardline_100,
    nxt.score_diff_post  AS next_score_diff,
    nxt.seconds_left_in_half AS next_seconds_left_in_half,
    nxt.posteam          AS next_posteam,
    nxt.timeout          AS next_timeout,
    nxt.game_half        AS next_game_half,
    nxt.posteam_timeouts_remaining AS next_posteam_timeouts_remaining,
    nxt.defteam_timeouts_remaining AS next_defteam_timeouts_remaining
  FROM current_state cur
  LEFT JOIN current_state nxt
    ON  nxt.season  = cur.season
    AND nxt.game_id = cur.game_id
    AND nxt.rn      = cur.rn + 1
    AND nxt.drive   <= cur.next_scoring_drive
),

-- Find the last play in each game that flips score_diff from negative to positive in the second half
lead_change_plays AS (
  SELECT
    season,
    game_id,
    MAX(rn) AS lead_change_rn   -- last such play per game
  FROM joined
  WHERE ((game_half = 'Half2') OR (game_half = 'Overtime'))
    AND score_diff < 0
    AND score_diff_post > 0
  GROUP BY season, game_id
),

q_dataset AS (
  SELECT
    j.play_id, j.drive, j.season_type, j.game_id, j.game_date, j.season, j.posteam, j.defteam, j.down, j.ydstogo/10.0 as ydstogo, j.yardline_100/100.0 as yardline_100, j.score_diff/8.0 as score_diff,
    CASE WHEN (j.game_half = 'Half1' or j.game_half = 'Half2') THEN j.seconds_left_in_half/1800.0 
    WHEN (j.game_half = 'Overtime' AND (j.season_type = 'POST' OR j.season < 2017)) THEN  j.seconds_left_in_half/900.0 
    ELSE  j.seconds_left_in_half/600.0 END as seconds_left_in_half,
    CASE WHEN j.game_half = 'Half1' THEN 0
         WHEN j.game_half = 'Half2' THEN 1
         ELSE 2 END AS game_half,
    CASE WHEN j.play_type = 'pass' THEN 0
         WHEN j.play_type = 'run'  THEN 1
         WHEN j.play_type = 'punt' THEN 2
         ELSE 3 END AS play_type,
    CASE WHEN j.down = 1 THEN 1 ELSE 0 END as first_down, 
    CASE WHEN j.down = 2 THEN 1 ELSE 0 END as second_down, 
    CASE WHEN j.down = 3 THEN 1 ELSE 0 END as third_down, 
    CASE WHEN j.down = 4 THEN 1 ELSE 0 END as fourth_down, 
    j.posteam_timeouts_remaining/3.0 as posteam_timeouts_remaining,
    j.defteam_timeouts_remaining/3.0 as defteam_timeouts_remaining,
    j.play_type AS action,
    -- Add +100 bonus if this is the last negative-to-positive flip in the second half
    j.reward, --+ CASE WHEN lcp.lead_change_rn IS NOT NULL THEN 9.0/(1+EXP(-0.02 * (120.0 - j.seconds_left_in_half))) ELSE 0 END AS reward,
    CASE WHEN j.next_down = 1 THEN 1 ELSE 0 END as next_first_down, 
    CASE WHEN j.next_down = 2 THEN 1 ELSE 0 END as next_second_down, 
    CASE WHEN j.next_down = 3 THEN 1 ELSE 0 END as next_third_down, 
    CASE WHEN j.next_down = 4 THEN 1 ELSE 0 END as next_fourth_down, 
    
    j.next_play_id, j.next_drive, j.next_down, j.next_ydstogo/10.0 as next_ydstogo, j.next_yardline_100/100.0 as next_yardline_100, j.next_score_diff/8.0 as next_score_diff,
    CASE WHEN (j.next_game_half = 'Half1' or j.next_game_half = 'Half2') THEN j.next_seconds_left_in_half/1800.0 
    WHEN (j.next_game_half = 'Overtime' AND (j.season_type = 'POST' OR j.season < 2017)) THEN  j.next_seconds_left_in_half/900.0 
    ELSE  j.next_seconds_left_in_half/600.0 END as next_seconds_left_in_half, j.next_posteam_timeouts_remaining/3.0 as next_posteam_timeouts_remaining, j.next_defteam_timeouts_remaining/3.0 as next_defteam_timeouts_remaining,
    CASE WHEN j.next_game_half = 'Half1' THEN 0
         WHEN j.next_game_half = 'Half2' THEN 1
         ELSE 2 END AS next_game_half,
    CASE WHEN j.next_posteam = j.posteam THEN 1 ELSE -1 END AS change_of_possession,
    CASE WHEN j.next_down IS NULL AND j.next_ydstogo IS NULL THEN 1 ELSE 0 END AS terminal_play
  FROM joined j
  LEFT JOIN lead_change_plays lcp
    ON  lcp.season       = j.season
    AND lcp.game_id      = j.game_id
    AND lcp.lead_change_rn = j.rn
)

SELECT *
FROM q_dataset
WHERE COLUMNS(c -> c IN ('down', 'ydstogo', 'yardline_100','score_diff','seconds_left_in_half','game_half','play_type','reward')) IS NOT NULL;