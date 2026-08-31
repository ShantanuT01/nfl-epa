import duckdb

if __name__ == "__main__":
    with open("sql/get_labels.sql") as f:
        duckdb.sql(f.read()).pl().write_parquet("data/ep_labels.parquet")

    with open("sql/get_games.sql") as f:
        duckdb.sql(f.read()).pl().write_parquet("data/elo.parquet")

    with open("sql/get_offline_rl.sql") as f:
        duckdb.sql(f.read()).pl().write_parquet("data/offline_rl.parquet")