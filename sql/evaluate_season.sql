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
    FROM ep.evaluations dqn

    LEFT JOIN ep.ep_labels ep
      USING (play_id, game_id, season)


  ),

  epas_per_play_and_seed AS (
    SELECT
      l.seed,
      l.game_id,
      l.play_id,
      l.next_play_id,
      l.season,
      l.posteam,
      l.defteam,
    CASE
      WHEN l.next_play_id IS NULL THEN 0                              -- genuine terminal: V(s') = 0
      WHEN r.play_id IS NULL THEN NULL                                -- next play expected but missing: flag, don't fabricate 0
      WHEN r.posteam = l.posteam THEN r.EP_epsilon
      ELSE -r.EP_epsilon END                                AS ep_after,
      CASE WHEN l.action = 'run'        THEN l.run
           WHEN l.action = 'pass'       THEN l.pass
           WHEN l.action = 'field_goal' THEN l.field_goal
           ELSE l.punt END                               AS qsa,
      l.EP_epsilon                                               AS ep_before,
     l.reward + ep_after - ep_before                 AS epa,
      l.next_scoring_drive,
      qsa - ep_before                                    AS advantage,
      l.episode_points,
      GREATEST(l.pass, l.run, l.punt, l.field_goal) - qsa AS opportunity_cost,
      l.reward + 0.99 * ep_after - ep_before              AS td_error,
      qsa - CASE GREATEST(l.pass_prob, l.punt_prob, l.field_goal_prob, l.run_prob)
               WHEN l.pass_prob       THEN l.pass
               WHEN l.run_prob        THEN l.run
               WHEN l.punt_prob       THEN l.punt
               ELSE l.field_goal END                     AS policy_gap,
      GREATEST(l.pass, l.run, l.punt, l.field_goal)  - ep_before AS regret,
      regret * advantage as weighted_advantage,
      CASE GREATEST(l.pass, l.run, l.punt, l.field_goal)
           WHEN l.pass       THEN 'pass'
           WHEN l.run        THEN 'run'
           WHEN l.punt       THEN 'punt'
           ELSE 'field_goal'
      END                                                AS greedy_action,
      l.action,

      -- Cross-entropy of team's action against DQN's probability distribution.
      -- = -log(p(chosen action)) where p comes from the DQN's softmax outputs.
      -- Higher = team choice was more surprising to the DQN (less aligned / more unpredictable).
      -- Lower  = team choice was more expected by the DQN (more aligned).
      -- NULLIF guards against log(0) on degenerate probability outputs.
      -LN(NULLIF(
        CASE l.action
          WHEN 'pass'       THEN l.pass_prob
          WHEN 'run'        THEN l.run_prob
          WHEN 'punt'       THEN l.punt_prob
          WHEN 'field_goal' THEN l.field_goal_prob
        END, 0)
      )                                                  AS action_cross_entropy

    FROM episodes l
    LEFT JOIN episodes r
      ON l.next_play_id = r.play_id
      AND r.game_id = l.game_id
      AND r.seed = l.seed
    --WHERE l.game_id = '2016_21_NE_ATL'
    WHERE l.season_type = 'REG'
  ),

  -- ==========================================================================
  -- Mean play EP: everything above this point is byte-for-byte what you
  -- wrote -- opportunity_cost, policy_gap, and greedy_action all still use
  -- each seed's own GREATEST() over that seed's own Q-values, untouched.
  --
  -- The only change is the V(s) baseline that epa / advantage / regret /
  -- weighted_advantage / td_error are computed against: instead of each
  -- seed's own EP_epsilon, they now use the ensemble mean of EP_epsilon
  -- across all 10 seeds for that specific play. This removes seed-to-seed
  -- model noise from the value baseline itself, the same way averaging an
  -- ensemble's predictions is supposed to.
  --
  -- mean_ep_before / mean_ep_after are computed with a window function
  -- (PARTITION BY game_id, play_id, no ORDER BY -- a plain full-partition
  -- average, not a running one) rather than a GROUP BY, so every per-seed
  -- row is preserved for opportunity_cost / policy_gap / greedy_action and
  -- for the seed-level STDDEV reporting further downstream.
  --
  -- epa / advantage / regret / td_error are re-derived from each metric's
  -- own existing definition rather than re-reading raw reward or Q columns,
  -- so epas_per_play_and_seed above never had to expose a new column:
  --   reward       = epa    - ep_after + ep_before   (epa's own definition)
  --   best_q_legal = regret + ep_before              (regret's own definition)
  -- Substituting the mean baseline into those same definitions gives the
  -- corrected values below. NULL handling: ep_after is NULL for a play
  -- whose next play is expected but missing (see the CASE above); AVG()
  -- ignores NULLs, so mean_ep_after is computed from whichever seeds have
  -- a value, and is NULL only if every seed is NULL for that play, same as
  -- before.
  -- ==========================================================================
  epas_with_mean_ep AS (
    SELECT
      *,
      AVG(ep_before) OVER (PARTITION BY game_id, play_id) AS mean_ep_before,
      AVG(ep_after)  OVER (PARTITION BY game_id, play_id) AS mean_ep_after
    FROM epas_per_play_and_seed
  ),

  epas AS (
    SELECT
      seed,
      game_id,
      play_id,
      next_play_id,
      season,
      posteam,
      defteam,
      next_scoring_drive,
      episode_points,
      action,
      action_cross_entropy,

      -- Untouched fundamental metrics -- passed through exactly as computed
      -- above, from each seed's own Q-values.
      qsa,
      opportunity_cost,
      policy_gap,
      greedy_action,

      -- Mean play EP: the ensemble-consensus V(s) baseline for this play.
      mean_ep_before                                                       AS ep_before,
      mean_ep_after                                                        AS ep_after,

      (epa - ep_after + ep_before) + mean_ep_after - mean_ep_before        AS epa,
      qsa - mean_ep_before                                                 AS advantage,
      (regret + ep_before) - mean_ep_before                                AS regret,
      ((regret + ep_before) - mean_ep_before) * (qsa - mean_ep_before)     AS weighted_advantage,
      (epa - ep_after + ep_before) + 0.99 * mean_ep_after - mean_ep_before AS td_error

    FROM epas_with_mean_ep
  ),

  offenses AS (
    SELECT
      seed,
      season,
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
      SUM(weighted_advantage)/SUM(regret)                    AS avg_weighted_advantage,
      AVG(regret)                                         AS regret,
      COUNT(DISTINCT game_id)                            AS games_played,
      AVG(CASE WHEN greedy_action = action THEN 1 ELSE 0 END) * 100 AS offense_agreement,
      AVG(CASE WHEN action = 'pass' THEN CASE WHEN greedy_action = action THEN 1 ELSE 0 END END) * 100 AS pass_agreement,
      AVG(CASE WHEN action = 'run'  THEN CASE WHEN greedy_action = action THEN 1 ELSE 0 END END) * 100 AS run_agreement,
      AVG(CASE WHEN action = 'punt' THEN CASE WHEN greedy_action = action THEN 1 ELSE 0 END END) * 100 AS punt_agreement,
      AVG(CASE WHEN action = 'field_goal' THEN CASE WHEN greedy_action = action THEN 1 ELSE 0 END END) * 100 AS fg_agreement,

      -- Play counts per action category — used for null-safe balanced agreement.
      -- A category with zero qualifying plays produces a NULL agreement value;
      -- these counts let us exclude those categories from the balanced average
      -- rather than treating NULL as zero or propagating it to the final metric.
      COUNT(CASE WHEN action = 'pass'       THEN 1 END)  AS n_pass_agreement_plays,
      COUNT(CASE WHEN action = 'run'        THEN 1 END)  AS n_run_agreement_plays,
      COUNT(CASE WHEN action = 'punt'       THEN 1 END)  AS n_punt_agreement_plays,
      COUNT(CASE WHEN action = 'field_goal' THEN 1 END)  AS n_fg_agreement_plays,

      AVG(CASE WHEN action = 'run'  THEN epa END)        AS off_epa_per_play_run,
      AVG(CASE WHEN action = 'pass' THEN epa END)        AS off_epa_per_play_pass,
      SUM(CASE WHEN action = 'run'  THEN epa END)        AS off_cumulative_epa_run,
      SUM(CASE WHEN action = 'pass' THEN epa END)        AS off_cumulative_epa_pass,
      COUNT(CASE WHEN action = 'run'        THEN 1 END)  AS n_runs,
      COUNT(CASE WHEN action = 'pass'       THEN 1 END)  AS n_passes,
      COUNT(CASE WHEN action = 'punt'       THEN 1 END)  AS n_punts,
      COUNT(CASE WHEN action = 'field_goal' THEN 1 END)  AS n_fgs,
      ROUND(COUNT(CASE WHEN action = 'pass' THEN 1 END)::FLOAT / COUNT(*), 3) AS pass_rate,
      AVG(CASE WHEN epa > 0 THEN 1 ELSE 0 END)           AS offense_success_rate,

      -- Per-action advantage: Q(s,a) - V(s) stratified by action type.
      -- Used to compute avg_advantage_balanced, which corrects for passing frequency
      -- bias in the overall avg_advantage metric in the same way offense_agreement_balanced
      -- corrects for bias in offense_agreement.
      AVG(CASE WHEN action = 'pass'       THEN advantage END) AS avg_advantage_pass,
      AVG(CASE WHEN action = 'run'        THEN advantage END) AS avg_advantage_run,
      AVG(CASE WHEN action = 'punt'       THEN advantage END) AS avg_advantage_punt,
      AVG(CASE WHEN action = 'field_goal' THEN advantage END) AS avg_advantage_fg,

      -- Cross-entropy: avg -log(p(team action)) under DQN policy.
      -- Higher = team choices more surprising to DQN (less aligned / more unpredictable).
      -- Lower  = team choices more expected by DQN (more aligned).
      -- Uniform random baseline over 4 actions = -log(0.25) ≈ 1.386.
      AVG(action_cross_entropy)                          AS off_cross_entropy

    FROM epas
    GROUP BY seed, posteam, season
  ),

  defenses AS (
    SELECT
      seed,
      season,
      defteam                                            AS team,
      AVG(-epa)                                          AS def_epa_suppressed_per_play,
      SUM(-epa)                                          AS def_cumulative_epa_suppressed,
      AVG(td_error)                                      AS avg_td_error,
      SUM(td_error)                                      AS def_cumulative_td_error,
      AVG(epa)                                           AS epa_allowed_per_play,
      SUM(epa)                                           AS cumulative_epa_allowed,
      AVG(CASE WHEN epa < 0 THEN 1 ELSE 0 END)           AS defense_success_rate,
      COUNT(DISTINCT game_id)                            AS games_played,
      AVG(CASE WHEN greedy_action = action THEN 1 ELSE 0 END) * 100 AS defense_agreement,
      AVG(CASE WHEN action = 'pass' THEN CASE WHEN greedy_action = action THEN 1 ELSE 0 END END) * 100 AS opp_pass_agreement,
      AVG(CASE WHEN action = 'run'  THEN CASE WHEN greedy_action = action THEN 1 ELSE 0 END END) * 100 AS opp_run_agreement,
      AVG(CASE WHEN action = 'punt' THEN CASE WHEN greedy_action = action THEN 1 ELSE 0 END END) * 100 AS opp_punt_agreement,
      AVG(CASE WHEN action = 'field_goal' THEN CASE WHEN greedy_action = action THEN 1 ELSE 0 END END) * 100 AS opp_fg_agreement,

      -- Play counts for opposing offense — mirrors offensive null-safe logic.
      COUNT(CASE WHEN action = 'pass'       THEN 1 END)  AS n_opp_pass_agreement_plays,
      COUNT(CASE WHEN action = 'run'        THEN 1 END)  AS n_opp_run_agreement_plays,
      COUNT(CASE WHEN action = 'punt'       THEN 1 END)  AS n_opp_punt_agreement_plays,
      COUNT(CASE WHEN action = 'field_goal' THEN 1 END)  AS n_opp_fg_agreement_plays,

      AVG(CASE WHEN action = 'run'  THEN epa END)        AS epa_allowed_run,
      AVG(CASE WHEN action = 'pass' THEN epa END)        AS epa_allowed_pass,
      SUM(CASE WHEN action = 'run'  THEN epa END)        AS cumulative_epa_allowed_run,
      SUM(CASE WHEN action = 'pass' THEN epa END)        AS cumulative_epa_allowed_pass,
      COUNT(CASE WHEN action = 'run'        THEN 1 END)  AS n_runs_faced,
      COUNT(CASE WHEN action = 'pass'       THEN 1 END)  AS n_passes_faced,
      COUNT(CASE WHEN action = 'punt'       THEN 1 END)  AS n_punts_faced,
      COUNT(CASE WHEN action = 'field_goal' THEN 1 END)  AS n_fgs_faced,
      ROUND(COUNT(CASE WHEN action = 'pass' THEN 1 END)::FLOAT / COUNT(*), 3) AS pass_rate_faced,

      -- Cross-entropy for opposing offenses (from defensive perspective).
      -- Higher = opposing offense more surprising to DQN (less aligned / more unpredictable).
      -- Lower  = opposing offense more expected by DQN (more aligned).
      AVG(action_cross_entropy)                          AS opp_cross_entropy

    FROM epas
    GROUP BY seed, defteam, season
  ),

  combined_per_seed AS (
    SELECT
      o.seed,
      o.season,
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
      o.avg_weighted_advantage,
      o.offense_agreement,
      o.regret,
      o.cumulative_policy_gap,
      o.pass_agreement,
      o.run_agreement,
      o.punt_agreement,
      o.fg_agreement,
      o.n_pass_agreement_plays,
      o.n_run_agreement_plays,
      o.n_punt_agreement_plays,
      o.n_fg_agreement_plays,
      o.offense_success_rate,
      o.off_epa_per_play_run,
      o.off_epa_per_play_pass,
      o.off_cumulative_epa_run,
      o.off_cumulative_epa_pass,
      o.n_runs,
      o.n_passes,
      o.n_punts,
      o.n_fgs,
      o.pass_rate,
      o.off_cross_entropy,
      o.avg_advantage_pass,
      o.avg_advantage_run,
      o.avg_advantage_punt,
      o.avg_advantage_fg,
      d.epa_allowed_per_play,
      d.cumulative_epa_allowed,
      d.avg_td_error                                     AS def_avg_td_error,
      d.def_cumulative_td_error,
      d.def_epa_suppressed_per_play,
      d.def_cumulative_epa_suppressed,
      d.defense_agreement,
      d.defense_success_rate,
      d.opp_pass_agreement,
      d.opp_run_agreement,
      d.opp_punt_agreement,
      d.opp_fg_agreement,
      d.n_opp_pass_agreement_plays,
      d.n_opp_run_agreement_plays,
      d.n_opp_punt_agreement_plays,
      d.n_opp_fg_agreement_plays,
      d.epa_allowed_run,
      d.epa_allowed_pass,
      d.epa_allowed_per_play,
      d.cumulative_epa_allowed_run,
      d.cumulative_epa_allowed_pass,
      d.n_runs_faced,
      d.n_passes_faced,
      d.n_punts_faced,
      d.n_fgs_faced,

      d.pass_rate_faced,
      d.opp_cross_entropy,
      o.epa_per_play - d.epa_allowed_per_play            AS net_epa_per_play,
      o.cumulative_epa - d.cumulative_epa_allowed        AS net_cumulative_epa
    FROM offenses o
    JOIN defenses d USING (seed, season, team)
  ),

season_stats AS (
SELECT
  team,
  season,
  AVG(games_played)                                      AS games_played,
  COUNT(DISTINCT seed)                                   AS n_seeds,

  -- Offensive overall
  ROUND(AVG(off_epa_per_play), 3)                        AS off_epa_per_play,
  ROUND(STDDEV(off_epa_per_play), 4)                     AS off_epa_per_play_sd,
  ROUND(AVG(avg_advantage), 3)                           AS avg_advantage,
  ROUND(STDDEV(avg_advantage), 4)                        AS avg_advantage_sd,
  ROUND(AVG(avg_weighted_advantage), 3)                  AS avg_weighted_advantage,
    ROUND(AVG(regret), 3)                  AS avg_regret,
  -- Balanced advantage: equal weight per action category, correcting for passing
  -- frequency bias in avg_advantage. NULL categories (no plays of that type)
  -- are excluded from both numerator and denominator via COALESCE + NULLIF.
  ROUND(
    (
      COALESCE(AVG(avg_advantage_pass), 0)
      + COALESCE(AVG(avg_advantage_run), 0)
      + COALESCE(AVG(avg_advantage_punt), 0)
      + COALESCE(AVG(avg_advantage_fg), 0)
    ) / NULLIF(
      CASE WHEN AVG(avg_advantage_pass) IS NOT NULL THEN 1 ELSE 0 END
      + CASE WHEN AVG(avg_advantage_run) IS NOT NULL THEN 1 ELSE 0 END
      + CASE WHEN AVG(avg_advantage_punt) IS NOT NULL THEN 1 ELSE 0 END
      + CASE WHEN AVG(avg_advantage_fg) IS NOT NULL THEN 1 ELSE 0 END
    , 0), 3
  )                                                      AS avg_advantage_balanced,
  ROUND(AVG(avg_policy_gap), 3)                          AS policy_gap,
  ROUND(STDDEV(avg_policy_gap), 4)                       AS policy_gap_sd,
  ROUND(AVG(avg_opportunity_cost), 3)                    AS avg_opportunity_cost,
  ROUND(STDDEV(avg_opportunity_cost), 3)                 AS avg_opportunity_cost_sd,
  ROUND(AVG(off_cumulative_epa / games_played), 3)       AS off_cumulative_epa_per_game,
  ROUND(AVG(cumulative_opportunity_cost / games_played), 3) AS opportunity_cost_per_game,
  ROUND(AVG(cumulative_advantage / games_played), 3)     AS advantage_per_game,
  ROUND(AVG(offense_agreement), 3)                       AS offense_agreement,
  ROUND(STDDEV(offense_agreement), 4)                    AS offense_agreement_sd,
  ROUND(AVG(cumulative_policy_gap / games_played), 3)    AS policy_gap_per_game,

  -- Offensive agreement by bucket
  ROUND(AVG(pass_agreement), 3)                          AS pass_agreement,
  ROUND(STDDEV(pass_agreement), 4)                       AS pass_agreement_sd,
  ROUND(AVG(run_agreement), 3)                           AS run_agreement,
  ROUND(STDDEV(run_agreement), 4)                        AS run_agreement_sd,
  ROUND(AVG(punt_agreement), 3)                          AS punt_agreement,
  ROUND(STDDEV(punt_agreement), 4)                       AS punt_agreement_sd,
  ROUND(AVG(fg_agreement), 3)                            AS fg_agreement,
  ROUND(STDDEV(fg_agreement), 4)                         AS fg_agreement_sd,

  -- Null-safe balanced agreement: average across action categories that have
  -- sufficient observations (>= 5 plays), weighted equally regardless of
  -- action frequency. Corrects for DQN passing bias in overall offense_agreement.
  -- n_agreement_categories shows how many of the 4 buckets contributed;
  -- interpret with caution when < 4 (common in small playoff samples).
  -- Balanced agreement: equal weight per action category regardless of frequency.
  -- Corrects for DQN passing bias in overall offense_agreement.
  -- All categories with at least 1 qualifying play are included (no minimum threshold).
  -- NULL agreement values (zero plays in that category) are excluded from both
  -- numerator and denominator via COALESCE + NULLIF so they don't distort the average.
  ROUND(
    (
      COALESCE(AVG(pass_agreement), 0)
      + COALESCE(AVG(run_agreement), 0)
      + COALESCE(AVG(punt_agreement), 0)
      + COALESCE(AVG(fg_agreement), 0)
    ) / NULLIF(
      CASE WHEN AVG(pass_agreement) IS NOT NULL THEN 1 ELSE 0 END
      + CASE WHEN AVG(run_agreement) IS NOT NULL THEN 1 ELSE 0 END
      + CASE WHEN AVG(punt_agreement) IS NOT NULL THEN 1 ELSE 0 END
      + CASE WHEN AVG(fg_agreement) IS NOT NULL THEN 1 ELSE 0 END
    , 0), 3
  )                                                      AS offense_agreement_balanced,

  -- Number of action categories contributing to offense_agreement_balanced.
  -- < 4 means at least one category had zero qualifying plays in this sample.
  CASE WHEN AVG(pass_agreement) IS NOT NULL THEN 1 ELSE 0 END
  + CASE WHEN AVG(run_agreement) IS NOT NULL THEN 1 ELSE 0 END
  + CASE WHEN AVG(punt_agreement) IS NOT NULL THEN 1 ELSE 0 END
  + CASE WHEN AVG(fg_agreement) IS NOT NULL THEN 1 ELSE 0 END
                                                         AS n_off_agreement_categories,

  -- Offensive cross-entropy vs DQN policy.
  -- Higher = team choices more surprising to DQN (less aligned / more unpredictable).
  -- Lower  = team choices more expected by DQN (more aligned).
  -- Uniform random baseline over 4 actions = -log(0.25) ≈ 1.386.
  ROUND(AVG(off_cross_entropy), 4)                       AS off_cross_entropy,
  ROUND(STDDEV(off_cross_entropy), 4)                    AS off_cross_entropy_sd,

  -- Offensive run/pass split
  ROUND(AVG(off_epa_per_play_run), 3)                    AS off_epa_run,
  ROUND(STDDEV(off_epa_per_play_run), 4)                 AS off_epa_run_sd,
  ROUND(AVG(off_epa_per_play_pass), 3)                   AS off_epa_pass,
  ROUND(STDDEV(off_epa_per_play_pass), 4)                AS off_epa_pass_sd,
  ROUND(AVG(off_cumulative_epa_run), 3)                  AS off_cumulative_epa_run,
  ROUND(AVG(off_cumulative_epa_pass), 3)                 AS off_cumulative_epa_pass,
  ROUND(AVG(epa_allowed_run), 3) as def_epa_run,
  ROUND(AVG(epa_allowed_pass), 3) as def_epa_pass,
  ROUND(AVG(offense_success_rate), 3)                    AS offense_success_rate,
  ROUND(AVG(defense_success_rate), 3)                    AS defense_success_rate,
  ROUND(AVG(n_runs), 0)                                  AS n_runs,
  ROUND(AVG(n_passes), 0)                                AS n_passes,
  ROUND(AVG(n_punts), 0)                                 AS n_punts,
  ROUND(AVG(n_fgs), 0)                                   AS n_fgs,
  ROUND(AVG(pass_rate), 3)                               AS pass_rate,
  ROUND(AVG(epa_allowed_per_play), 3)                    AS epa_allowed_per_play,
  ROUND(STDDEV(epa_allowed_per_play), 4)                 AS epa_allowed_per_play_sd,
  ROUND(AVG(cumulative_epa_allowed / games_played), 3)   AS epa_allowed_per_game,
  ROUND(AVG(defense_agreement), 3)                       AS opposing_offense_agreement,
  ROUND(STDDEV(defense_agreement), 4)                    AS opposing_offense_agreement_sd,

  -- Opposing offense agreement by bucket
  ROUND(AVG(opp_pass_agreement), 3)                      AS opp_pass_agreement,
  ROUND(STDDEV(opp_pass_agreement), 4)                   AS opp_pass_agreement_sd,
  ROUND(AVG(opp_run_agreement), 3)                       AS opp_run_agreement,
  ROUND(STDDEV(opp_run_agreement), 4)                    AS opp_run_agreement_sd,
  ROUND(AVG(opp_punt_agreement), 3)                      AS opp_punt_agreement,
  ROUND(STDDEV(opp_punt_agreement), 4)                   AS opp_punt_agreement_sd,
  ROUND(AVG(opp_fg_agreement), 3)                        AS opp_fg_agreement,
  ROUND(STDDEV(opp_fg_agreement), 4)                     AS opp_fg_agreement_sd,

  -- Null-safe balanced agreement for opposing offense.
  -- Same logic as offense_agreement_balanced — corrects for passing bias,
  -- excludes action categories with < 5 qualifying plays.
  -- Balanced agreement for opposing offense — same logic as offense_agreement_balanced.
  ROUND(
    (
      COALESCE(AVG(opp_pass_agreement), 0)
      + COALESCE(AVG(opp_run_agreement), 0)
      + COALESCE(AVG(opp_punt_agreement), 0)
      + COALESCE(AVG(opp_fg_agreement), 0)
    ) / NULLIF(
      CASE WHEN AVG(opp_pass_agreement) IS NOT NULL THEN 1 ELSE 0 END
      + CASE WHEN AVG(opp_run_agreement) IS NOT NULL THEN 1 ELSE 0 END
      + CASE WHEN AVG(opp_punt_agreement) IS NOT NULL THEN 1 ELSE 0 END
      + CASE WHEN AVG(opp_fg_agreement) IS NOT NULL THEN 1 ELSE 0 END
    , 0), 3
  )                                                      AS opposing_offense_agreement_balanced,

  -- Number of action categories contributing to opposing_offense_agreement_balanced.
  CASE WHEN AVG(opp_pass_agreement) IS NOT NULL THEN 1 ELSE 0 END
  + CASE WHEN AVG(opp_run_agreement) IS NOT NULL THEN 1 ELSE 0 END
  + CASE WHEN AVG(opp_punt_agreement) IS NOT NULL THEN 1 ELSE 0 END
  + CASE WHEN AVG(opp_fg_agreement) IS NOT NULL THEN 1 ELSE 0 END
                                                         AS n_opp_agreement_categories,

  -- Opposing offense cross-entropy vs DQN policy.
  -- Higher = opposing offense more surprising to DQN (less aligned / more unpredictable).
  -- Lower  = opposing offense more expected by DQN (more aligned).
  ROUND(AVG(opp_cross_entropy), 4)                       AS opp_cross_entropy,
  ROUND(STDDEV(opp_cross_entropy), 4)                    AS opp_cross_entropy_sd,

  -- Defensive run/pass split
  ROUND(AVG(epa_allowed_run), 3)                         AS def_epa_allowed_run,
  ROUND(AVG(epa_allowed_pass), 3)                        AS def_epa_allowed_pass,
  ROUND(AVG(cumulative_epa_allowed_run), 3)              AS cumulative_epa_allowed_run,
  ROUND(AVG(cumulative_epa_allowed_pass), 3)             AS cumulative_epa_allowed_pass,
  ROUND(AVG(n_runs_faced), 0)                            AS n_runs_faced,
  ROUND(AVG(n_passes_faced), 0)                          AS n_passes_faced,
  ROUND(AVG(n_punts_faced), 0)                           AS n_punts_faced,
  ROUND(AVG(n_fgs_faced), 0)                             AS n_fgs_faced,
  ROUND(AVG(pass_rate_faced), 3)                         AS pass_rate_faced,

  -- Net metrics with variance
  ROUND(AVG(net_epa_per_play), 3)                        AS net_epa_per_play,
  ROUND(STDDEV(net_epa_per_play), 4)                     AS net_epa_per_play_sd,
  ROUND(AVG(net_cumulative_epa / games_played), 3)       AS net_epa_per_game,
  ROUND(STDDEV(net_cumulative_epa / games_played), 4)    AS net_epa_per_game_sd,

  -- advantage
  ROUND(AVG(avg_advantage_pass),3) AS avg_advantage_pass,
  ROUND(AVG(avg_advantage_run), 3) AS avg_advantage_run,
  ROUND(AVG(avg_advantage_punt), 3) AS avg_advantage_punt,
  ROUND(AVG(avg_advantage_fg), 3) AS avg_advantage_fg
FROM combined_per_seed
GROUP BY team, season
ORDER BY opp_cross_entropy DESC
)

SELECT
season,
  team,
  net_epa_per_play,
--  net_epa_per_play_sd,
    off_epa_per_play,
  epa_allowed_per_play,
  off_epa_run,

  off_epa_pass,
  def_epa_run,
  def_epa_pass,
  offense_agreement_balanced,
  n_off_agreement_categories,
  opposing_offense_agreement_balanced,
  n_opp_agreement_categories,
  pass_agreement,
  run_agreement,
  punt_agreement,
  fg_agreement,

  avg_advantage_sd,
  avg_advantage_balanced,
  avg_opportunity_cost,
  avg_opportunity_cost_sd,
  policy_gap,
  policy_gap_sd,
  avg_advantage_pass,
  avg_advantage_run,
  avg_advantage_punt,
  avg_advantage_fg,
  avg_weighted_advantage,
    avg_advantage,
  avg_regret
FROM season_stats
WHERE season = 2017
ORDER BY net_epa_per_play DESC;