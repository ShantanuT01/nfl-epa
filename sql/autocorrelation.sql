WITH
  episodes AS (
    SELECT
      dqn.*,
      ep.next_scoring_drive,
      CASE
        WHEN ep.next_score_label = 0 THEN -6
        WHEN ep.next_score_label = 1 THEN -3
        WHEN ep.next_score_label = 2 THEN -2
        WHEN ep.next_score_label = 3 THEN 0
        WHEN ep.next_score_label = 4 THEN 2
        WHEN ep.next_score_label = 5 THEN 3
        ELSE 6
      END AS episode_points
    FROM
      ep.evaluations dqn
      LEFT JOIN ep.ep_labels ep USING (play_id, game_id, season)
  ),
  -- compute metrics per play/seed
  epas_per_play_and_seed AS (
    SELECT
      l.seed,
      l.game_id,
      l.play_id,
      l.next_play_id,
      l.season,
      l.posteam,
      l.defteam,
      l.pass,
      l.run,
      l.punt,
      l.field_goal,
      l.pass_prob,
      l.run_prob,
      l.punt_prob,
      l.field_goal_prob,
      CASE
        WHEN l.next_play_id IS NULL THEN 0 -- genuine terminal: V(s') = 0
        WHEN r.play_id IS NULL THEN NULL -- next play expected but missing: flag, don't fabricate 0
        WHEN r.posteam = l.posteam THEN r.EP_epsilon
        ELSE - r.EP_epsilon
      END AS ep_after,
      CASE
        WHEN l.action = 'run' THEN l.run
        WHEN l.action = 'pass' THEN l.pass
        WHEN l.action = 'field_goal' THEN l.field_goal
        ELSE l.punt
      END AS qsa,
      l.EP_epsilon AS ep_before,
      l.reward + ep_after - ep_before AS epa,
      l.next_scoring_drive,
      qsa - ep_before AS advantage,
      l.episode_points,
      GREATEST(l.pass, l.run, l.punt, l.field_goal) - qsa AS opportunity_cost,
      l.reward + 0.99 * ep_after - ep_before AS td_error,
      qsa - CASE GREATEST(l.pass_prob, l.punt_prob, l.field_goal_prob, l.run_prob)
        WHEN l.pass_prob THEN l.pass
        WHEN l.run_prob THEN l.run
        WHEN l.punt_prob THEN l.punt
        ELSE l.field_goal
      END AS policy_gap,
      CASE GREATEST(l.pass, l.run, l.punt, l.field_goal)
        WHEN l.pass THEN 'pass'
        WHEN l.run THEN 'run'
        WHEN l.punt THEN 'punt'
        ELSE 'field_goal'
      END AS greedy_action,
      l.action,
    FROM
      episodes l
      LEFT JOIN episodes r ON l.next_play_id = r.play_id
      AND r.game_id = l.game_id
      AND r.seed = l.seed
  
      --WHERE l.game_id = '2016_21_NE_ATL'
      WHERE l.season_type = 'REG' 
  ),
  epas AS (
    SELECT
      season,
      game_id,
      play_id,
      posteam,
      defteam,
      AVG(ep_after) as ep_after,
      STDDEV(ep_after) as ep_after_sd,
      AVG(ep_before) AS ep_before,
      STDDEV(ep_before) as ep_before_sd,
      AVG(epa) as epa,
      STDDEV(ep_after - ep_before) as epa_sd,
      AVG(qsa - ep_before) as advantage,
      STDDEV(qsa - ep_before) as advantage_sd,
      AVG(opportunity_cost) AS opportunity_cost,
      STDDEV(opportunity_cost) AS opportunity_cost_sd,
      AVG(qsa) as qsa,
      STDDEV(qsa) as qsa_sd,
      SUM(
        CASE
          WHEN greedy_action = 'pass' THEN 1
          ELSE 0
        END
      ) / COUNT(*) as pass_recommendation,
      SUM(
        CASE
          WHEN greedy_action = 'run' THEN 1
          ELSE 0
        END
      ) / COUNT(*) as run_recommendation,
      SUM(
        CASE
          WHEN greedy_action = 'field_goal' THEN 1
          ELSE 0
        END
      ) / COUNT(*) as field_goal_recommendation,
      SUM(
        CASE
          WHEN greedy_action = 'punt' THEN 1
          ELSE 0
        END
      ) / COUNT(*) as punt_recommendation,
      AVG(pass) AS pass,
      STDDEV(pass) as pass_sd,
      AVG(run) AS run,
      STDDEV(run) as run_sd,
      AVG(punt) as punt,
      STDDEV(punt) as punt_sd,
      AVG(field_goal) as field_goal,
      STDDEV(field_goal) as field_goal_sd,
      AVG(pass_prob) as pass_prob,
      STDDEV(pass_prob) as pass_prob_sd,
      AVG(run_prob) as run_prob,
      STDDEV(run_prob) as run_prob_sd,
      AVG(punt_prob) as punt_prob,
      STDDEV(punt_prob) as punt_prob_sd,
      AVG(field_goal_prob) as field_goal_prob,
      STDDEV(field_goal_prob) AS field_goal_prob_sd,
      GREATEST(pass_recommendation, run_recommendation, punt_recommendation, field_goal_recommendation) AS consensus_rate,
-(
  CASE WHEN pass_recommendation       > 0 THEN pass_recommendation       * LN(pass_recommendation)       ELSE 0 END +
  CASE WHEN run_recommendation        > 0 THEN run_recommendation        * LN(run_recommendation)        ELSE 0 END +
  CASE WHEN punt_recommendation       > 0 THEN punt_recommendation       * LN(punt_recommendation)       ELSE 0 END +
  CASE WHEN field_goal_recommendation > 0 THEN field_goal_recommendation * LN(field_goal_recommendation) ELSE 0 END
) AS action_entropy,
    FROM epas_per_play_and_seed
    GROUP BY
      season,
      game_id,
      play_id,
      posteam,
      defteam
  ),
  offenses AS (
    SELECT
      season,
      posteam AS team,
      AVG(advantage) AS avg_advantage,
      AVG(epa) AS epa_per_play,
      AVG(opportunity_cost) AS avg_opportunity_cost,
      AVG(qsa) as avg_qsa, 
      COUNT(DISTINCT game_id) AS games_played,
      AVG(action_entropy) as avg_action_entropy,
      AVG(consensus_rate) as avg_consensus_rate
    FROM
      epas
    GROUP BY
      posteam,
      season
  ),
  defenses AS (
    SELECT
      season,
      defteam AS team,
      AVG(epa) AS epa_allowed_per_play,
      AVG(advantage) AS avg_advantage_allowed,
      AVG(opportunity_cost) AS avg_opportunity_cost_allowed,
      AVG(qsa) AS avg_qsa_allowed,
      AVG(action_entropy) AS avg_action_entropy_allowed,
      AVG(consensus_rate) AS avg_consensus_rate_allowed,
    FROM
      epas
    GROUP BY
      defteam,
      season
  ),
season_stats AS (
SELECT
  o.season,
  o.team,
  o.epa_per_play - d.epa_allowed_per_play AS net_epa_per_play,
  o.epa_per_play AS off_epa_per_play,
  d.epa_allowed_per_play as def_epa_per_play,
  o.avg_advantage,
  d.avg_advantage_allowed,
  o.avg_advantage -   d.avg_advantage_allowed as net_advantage,

  o.avg_qsa,

  d.avg_qsa_allowed,
  o.avg_qsa - d.avg_qsa_allowed as net_qsa, 
  o.avg_opportunity_cost,
  d.avg_opportunity_cost_allowed,
  d.avg_opportunity_cost_allowed  - o.avg_opportunity_cost as net_opportunity_cost,
  o.avg_action_entropy,
  d.avg_action_entropy_allowed,
  o.avg_consensus_rate,
  d.avg_consensus_rate_allowed
  

FROM
  offenses o
  JOIN defenses d USING (season, team)
  
),
paired AS (
  SELECT
    a.team,
    a.season       AS season_t,
    a.avg_advantage   AS advantage_t,
    b.avg_advantage   AS advantage_t1,
    a.avg_opportunity_cost AS oc_t,
    b.avg_opportunity_cost AS oc_t1,
    a.epa_per_play     AS epa_t,
    b.epa_per_play     AS epa_t1
  FROM offenses a
  JOIN offenses b
    ON a.team = b.team
   AND b.season = a.season + 1
)
SELECT * 
FROM paired;