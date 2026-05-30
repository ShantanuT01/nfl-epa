import duckdb
with open("sql/get_labels.sql") as f:
    duckdb.sql(f.read()).pl().write_parquet("ep_labels.parquet")

with open("sql/get_games.sql") as f:
    duckdb.sql(f.read()).pl().write_parquet("elo.parquet")