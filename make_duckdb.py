import duckdb


if __name__ == "__main__":
    with duckdb.connect("ep_db.duckdb") as con:
        # with open("sql/duckdb/get_policy_predictions.sql") as f:
        #     con.execute(f.read())

        # with open("sql/duckdb/get_raw_pbp.sql") as f:
        #     con.execute(f.read())
        
        # with open("sql/duckdb/get_season_values.sql") as f:
        #     con.execute(f.read())

        # with open("sql/duckdb/get_ep_labels.sql") as f:
        #     con.execute(f.read())

        with open("sql/duckdb/get_games.sql") as f:
            con.execute(f.read())