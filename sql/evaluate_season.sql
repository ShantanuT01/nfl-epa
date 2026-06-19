WITH
  episodes AS (
    SELECT 
      dqn.*,
      ep.next_scoring_drive,
      CASE WHEN ep.next_score_label = 0 THEN -6
           WHEN ep.next_score_label = 1 THEN -3
           WHEN ep.next_score_label = 2 THEN -2
           WHEN ep.next_score_label = 3 THEN 0
           WHEN ep.next_score_label = 4 THEN 2
           WHEN ep.next_score_label = 5 THEN 3
           ELSE 6 END AS episode_points
    FROM 'evaluations/2025/*.parquet' dqn
    LEFT JOIN 'data/ep_labels.parquet' ep
      ON ep.play_id = dqn.play_id 
      AND ep.game_id = dqn.game_id
  ),

  epas AS (
    SELECT
      l.game_id, 
      l.play_id,
      l.next_play_id,
      l.season,
      l.posteam,
      l.defteam,
      CASE WHEN r.posteam = l.posteam THEN r.ep
           ELSE -r.ep END                                AS ep_after,
      CASE WHEN l.action = 'run'        THEN l.run 
           WHEN l.action = 'pass'       THEN l.pass 
           WHEN l.action = 'field_goal' THEN l.field_goal
           ELSE l.punt END                               AS qsa, 
      l.ep                                               AS ep_before,
      l.reward + ep_after - ep_before                    AS epa,
      l.next_scoring_drive, 
      qsa - ep_before                                    AS advantage,
      l.episode_points,
      GREATEST(l.pass, l.run, l.punt, l.field_goal) - qsa AS opportunity_cost,
      l.reward + 0.8 * ep_after - ep_before              AS td_error,
      qsa - CASE GREATEST(l.pass_prob, l.punt_prob, l.field_goal_prob, l.run_prob) 
               WHEN l.pass_prob       THEN l.pass 
               WHEN l.run_prob        THEN l.run 
               WHEN l.punt_prob       THEN l.punt
               ELSE l.field_goal END                     AS policy_gap,
      CASE GREATEST(l.pass, l.run, l.punt, l.field_goal)
           WHEN l.pass       THEN 'pass'
           WHEN l.run        THEN 'run'
           WHEN l.punt       THEN 'punt'
           ELSE 'field_goal'
      END                                                AS greedy_action,
      l.action
    FROM episodes l
    LEFT JOIN episodes r 
      ON l.next_play_id = r.play_id 
      AND r.game_id = l.game_id
    --WHERE l.game_id = '2017_19_NO_MIN' 
  ),

  offenses AS (
    SELECT 
      posteam                                            AS team,
      AVG(advantage)                                     AS avg_advantage, 
      AVG(td_error)                                      AS avg_td_error, 
      AVG(policy_gap)                                    AS avg_policy_gap,
      SUM(policy_gap)                                    AS cumulative_policy_gap, 
      AVG(epa)                                           AS epa_per_play,
      AVG(opportunity_cost)                              AS avg_opportunity_cost,
      SUM(epa)                                           AS cumulative_epa, 
      SUM(opportunity_cost)                              AS cumulative_opportunity_cost,
      SUM(advantage)                                     AS cumulative_advantage,
      COUNT(DISTINCT game_id)                            AS games_played,

      -- Overall agreement
      AVG(CASE WHEN greedy_action = action 
               THEN 1 ELSE 0 END) * 100                 AS offense_agreement,

      -- Agreement by action bucket
      AVG(CASE WHEN action = 'pass'
               THEN CASE WHEN greedy_action = action 
                         THEN 1 ELSE 0 END
          END) * 100                                     AS pass_agreement,
      AVG(CASE WHEN action = 'run'
               THEN CASE WHEN greedy_action = action 
                         THEN 1 ELSE 0 END
          END) * 100                                     AS run_agreement,
      AVG(CASE WHEN action = 'punt'
               THEN CASE WHEN greedy_action = action 
                         THEN 1 ELSE 0 END
          END) * 100                                     AS punt_agreement,
      AVG(CASE WHEN action = 'field_goal'
               THEN CASE WHEN greedy_action = action 
                         THEN 1 ELSE 0 END
          END) * 100                                     AS fg_agreement,

      -- Offensive run/pass split
      AVG(CASE WHEN action = 'run'  THEN epa END)        AS off_epa_per_play_run,
      AVG(CASE WHEN action = 'pass' THEN epa END)        AS off_epa_per_play_pass,
      SUM(CASE WHEN action = 'run'  THEN epa END)        AS off_cumulative_epa_run,
      SUM(CASE WHEN action = 'pass' THEN epa END)        AS off_cumulative_epa_pass,
      COUNT(CASE WHEN action = 'run'        THEN 1 END)  AS n_runs,
      COUNT(CASE WHEN action = 'pass'       THEN 1 END)  AS n_passes,
      COUNT(CASE WHEN action = 'punt'       THEN 1 END)  AS n_punts,
      COUNT(CASE WHEN action = 'field_goal' THEN 1 END)  AS n_fgs,
      ROUND(COUNT(CASE WHEN action = 'pass' THEN 1 END)::FLOAT 
            / COUNT(*), 3)                               AS pass_rate,
      AVG(CASE WHEN epa > 0 THEN 1 ELSE 0 END) as offense_success_rate, 
    FROM epas
    GROUP BY posteam
  ),

  defenses AS (
    SELECT 
      defteam                                            AS team,
      AVG(-epa)                                          AS def_epa_suppressed_per_play,
      SUM(-epa)                                          AS def_cumulative_epa_suppressed,
      AVG(td_error)                                      AS def_avg_td_error,
      SUM(td_error)                                      AS def_cumulative_td_error,
      AVG(epa)                                           AS epa_allowed_per_play,
      SUM(epa)                                           AS cumulative_epa_allowed,
      AVG(CASE WHEN epa < 0 THEN 1 ELSE 0 END) as defense_success_rate, 
      COUNT(DISTINCT game_id)                            AS games_played,

      -- Overall opposing offense agreement
      AVG(CASE WHEN greedy_action = action 
               THEN 1 ELSE 0 END) * 100                 AS defense_agreement,

      -- Opposing offense agreement by action bucket
      AVG(CASE WHEN action = 'pass'
               THEN CASE WHEN greedy_action = action 
                         THEN 1 ELSE 0 END
          END) * 100                                     AS opp_pass_agreement,
      AVG(CASE WHEN action = 'run'
               THEN CASE WHEN greedy_action = action 
                         THEN 1 ELSE 0 END
          END) * 100                                     AS opp_run_agreement,
      AVG(CASE WHEN action = 'punt'
               THEN CASE WHEN greedy_action = action 
                         THEN 1 ELSE 0 END
          END) * 100                                     AS opp_punt_agreement,
      AVG(CASE WHEN action = 'field_goal'
               THEN CASE WHEN greedy_action = action 
                         THEN 1 ELSE 0 END
          END) * 100                                     AS opp_fg_agreement,

      -- Defensive run/pass split
      AVG(CASE WHEN action = 'run'  THEN epa END)        AS epa_allowed_run,
      AVG(CASE WHEN action = 'pass' THEN epa END)        AS epa_allowed_pass,
      SUM(CASE WHEN action = 'run'  THEN epa END)        AS cumulative_epa_allowed_run,
      SUM(CASE WHEN action = 'pass' THEN epa END)        AS cumulative_epa_allowed_pass,
      COUNT(CASE WHEN action = 'run'        THEN 1 END)  AS n_runs_faced,
      COUNT(CASE WHEN action = 'pass'       THEN 1 END)  AS n_passes_faced,
      COUNT(CASE WHEN action = 'punt'       THEN 1 END)  AS n_punts_faced,
      COUNT(CASE WHEN action = 'field_goal' THEN 1 END)  AS n_fgs_faced,
      ROUND(COUNT(CASE WHEN action = 'pass' THEN 1 END)::FLOAT 
            / COUNT(*), 3)                               AS pass_rate_faced
    FROM epas
    GROUP BY defteam
  ),

  combined AS (
    SELECT
      o.team,
      o.games_played,
      o.avg_advantage,
      o.avg_td_error                                     AS off_avg_td_error,
      o.avg_policy_gap,
      o.epa_per_play                                     AS off_epa_per_play,
      o.avg_opportunity_cost,
      o.cumulative_epa                                   AS off_cumulative_epa,
      o.cumulative_opportunity_cost,
      o.cumulative_advantage,
      o.offense_agreement,
      o.cumulative_policy_gap,
      -- Offensive agreement by bucket
      o.pass_agreement,
      o.run_agreement,
      o.punt_agreement,
      o.offense_success_rate, 
      o.fg_agreement,
      -- Offensive run/pass split
      o.off_epa_per_play_run,
      o.off_epa_per_play_pass,
      o.off_cumulative_epa_run,
      o.off_cumulative_epa_pass,
      o.n_runs,
      o.n_passes,
      o.n_punts,
      o.n_fgs,
      o.pass_rate,
      -- Defensive metrics
      d.epa_allowed_per_play,
      d.cumulative_epa_allowed,
      d.def_avg_td_error,
      d.def_cumulative_td_error,
      d.def_epa_suppressed_per_play,
      d.def_cumulative_epa_suppressed,
      d.defense_agreement,
      d.defense_success_rate, 
      -- Opposing offense agreement by bucket
      d.opp_pass_agreement,
      d.opp_run_agreement,
      d.opp_punt_agreement,
      d.opp_fg_agreement,
      -- Defensive run/pass split
      d.epa_allowed_run,
      d.epa_allowed_pass,
      d.cumulative_epa_allowed_run,
      d.cumulative_epa_allowed_pass,
      d.n_runs_faced,
      d.n_passes_faced,
      d.n_punts_faced,
      d.n_fgs_faced,
      d.pass_rate_faced,
      -- Net metrics
      ROUND(o.epa_per_play - d.epa_allowed_per_play, 3) AS net_epa_per_play,
      ROUND(o.cumulative_epa - d.cumulative_epa_allowed, 3) AS net_cumulative_epa
    FROM offenses o
    JOIN defenses d ON o.team = d.team
  ),
subset AS(
SELECT
  team,
  games_played,
  -- Offensive overall
  ROUND(off_epa_per_play, 3)                             AS off_epa_per_play,
  ROUND(avg_advantage, 3)                                AS avg_advantage,
  ROUND(avg_policy_gap, 3)                               AS policy_gap,
  ROUND(avg_opportunity_cost, 3)                         AS avg_opportunity_cost,
  ROUND(off_cumulative_epa / games_played, 3)            AS off_cumulative_epa_per_game,
  ROUND(cumulative_opportunity_cost / games_played, 3)   AS opportunity_cost_per_game,
  ROUND(cumulative_advantage / games_played, 3)          AS advantage_per_game,
  ROUND(offense_agreement, 3)                            AS offense_agreement,
  ROUND(cumulative_policy_gap / games_played, 3)         AS policy_gap_per_game,
  -- Offensive agreement by bucket
  ROUND(pass_agreement, 3)                               AS pass_agreement,
  ROUND(run_agreement, 3)                                AS run_agreement,
  ROUND(punt_agreement, 3)                               AS punt_agreement,
  ROUND(fg_agreement, 3)                                 AS fg_agreement,
  -- Offensive run/pass split
  ROUND(off_epa_per_play_run, 3)                         AS off_epa_run,
  ROUND(off_epa_per_play_pass, 3)                        AS off_epa_pass,
  ROUND(off_cumulative_epa_run, 3)                       AS off_cumulative_epa_run,
  ROUND(off_cumulative_epa_pass, 3)                      AS off_cumulative_epa_pass,
  ROUND(offense_success_rate, 3) AS offense_success_rate, 
  ROUND(defense_success_rate, 3) AS defense_success_rate, 
  n_runs,
  n_passes,
  n_punts,
  n_fgs,
  ROUND(pass_rate, 3)                                    AS pass_rate,
  -- Defensive overall
  ROUND(epa_allowed_per_play, 3)                         AS epa_allowed_per_play,
  ROUND(cumulative_epa_allowed / games_played, 3)        AS epa_allowed_per_game,
  ROUND(defense_agreement, 3)                            AS opposing_offense_agreement,
  -- Opposing offense agreement by bucket
  ROUND(opp_pass_agreement, 3)                           AS opp_pass_agreement,
  ROUND(opp_run_agreement, 3)                            AS opp_run_agreement,
  ROUND(opp_punt_agreement, 3)                           AS opp_punt_agreement,
  ROUND(opp_fg_agreement, 3)                             AS opp_fg_agreement,
  -- Defensive run/pass split
  ROUND(epa_allowed_run, 3)                              AS def_epa_allowed_run,
  ROUND(epa_allowed_pass, 3)                             AS def_epa_allowed_pass,
  ROUND(cumulative_epa_allowed_run, 3)                   AS cumulative_epa_allowed_run,
  ROUND(cumulative_epa_allowed_pass, 3)                  AS cumulative_epa_allowed_pass,
  n_runs_faced,
  n_passes_faced,
  n_punts_faced,
  n_fgs_faced,
  ROUND(pass_rate_faced, 3)                              AS pass_rate_faced,
  -- Net
  net_epa_per_play,
  ROUND(net_cumulative_epa / games_played, 3)            AS net_epa_per_game
FROM combined
ORDER BY net_epa_per_play DESC)
SELECT *
FROM subset
ORDER BY net_epa_per_play DESC