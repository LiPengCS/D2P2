import pandas as pd
import os
import json

datasets = [d for d in os.listdir("uci10") if os.path.isdir(os.path.join("uci10", d))]

summary = []
for d in datasets:
    data = pd.read_csv(os.path.join("uci10", d, "data.csv"), na_values="?")
    num_mv = data.isnull().values.mean()
    summary.append([d, data.shape[0], data.shape[1], num_mv])

summary = pd.DataFrame(summary, columns=["dataset", "# examples", "# features", "% mv"])
summary.to_csv('summary_uci10.csv', index=False)