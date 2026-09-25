import yaml
import os 

if __name__ == "__main__":
    with open("configs/2016.yaml",'r') as f:
        data = yaml.safe_load(f)
    
    for year in range(2017, 2026):
        data["train_start_season"] = year - 11
        data["train_end_season"] = year - 1
        data["test_season"] = year
        data["checkpoint_path"] = f"models/{year}/"
        with open(f"configs/{year}.yaml",'w+') as f:
            yaml.dump(data, f, sort_keys=False)
        
