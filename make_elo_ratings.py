import pandas as pd
from collections import defaultdict
df = pd.read_parquet("elo.parquet")

df["pregame_home_elo"] = None
df["postgame_home_elo"] = None
df["pregame_away_elo"] = None
df["postgame_away_elo"] = None

for idx, row in df.iterrows():
    home_team = row["home_team"]
    away_team = row["away_team"]
    week, season = row["week"], row["season"]
    away_score = row["away_score"]
    home_score = row["home_score"]
    # in week 1 perform reversion
    if week == 1:
        if season == 2002:
            df.at[idx, "pregame_home_elo"] = 1300
            df.at[idx, "pregame_away_elo"] = 1300
        else:
            last_season = pd.DataFrame(df[df.season == season - 1])
            #last_season[(last_season.home_team == home_team) | (last_season.away_team == home_team)]
            #df.at[idx, "pregame_home_elo"] = 1300
            #df.at[idx, "pregame_away_elo"] = 1300
    # else get previous game

