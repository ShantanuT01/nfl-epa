from ep import EPModel, StandardEPModel, EPDataset
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
import lightning as L
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    df = pd.read_parquet("data/ep_labels.parquet")
    #df["ydstogo"] = np.log(df["ydstogo"] + 1)
    #df["yardline_100"] = np.log(df["yardline_100"] + 1)
    #df["score_diff"]/=8
    df["down"] -=  1
    df["qtr"] -= 1
    df["posteam"] = df['posteam'].astype('category').cat.codes
    df["defteam"] = df['defteam'].astype('category').cat.codes
    df["next_score_label"] = (df["next_score_label"] == 0).astype(int)
    #df["seconds_left_in_half"] = np.log(df["seconds_left_in_half"] + 1)
    df = df[["season","down","ydstogo","yardline_100","home_advantage", "score_diff","seconds_left_in_half","posteam","defteam","qtr"] +["next_score_label"]].dropna()
    train_df = df[df.season.between(2016, 2019)]
    test_df = df[df.season == 2020]
    train_cols = ["ydstogo","yardline_100","score_diff","seconds_left_in_half", "down","home_advantage","qtr"]
    
    X_train = train_df[train_cols].to_numpy()
    y_train = train_df["next_score_label"].to_numpy(dtype=int)
    train_dataset = EPDataset(X_train, y_train)
    X_test = test_df[train_cols].to_numpy()
    y_test = test_df["next_score_label"].to_numpy(dtype=int)
    
    test_dataset = EPDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True,num_workers=4)
    val_loader = DataLoader(test_dataset, batch_size=128, shuffle=False,num_workers=4)
    model = EPModel(10, 4, {"down": 4, "home_advantage": 2,"qtr":5}, 128)
    model_wrapper = StandardEPModel(model,"mps")
    trainer = L.Trainer(max_epochs=2, accelerator='mps', detect_anomaly=True)

    trainer.fit(model_wrapper, train_dataloaders=train_loader)
    output = trainer.predict(model=model_wrapper, dataloaders=val_loader)
    y_score = torch.cat(output).numpy()#.reshape(len(y_test),7)
    print(roc_auc_score(y_score=y_score, y_true=y_test))