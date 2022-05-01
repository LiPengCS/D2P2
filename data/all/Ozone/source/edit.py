import pandas as pd

data = pd.read_csv("onehr.csv")

feature_name = []
with open("feature_name.txt") as f:
    for line in f:
        words = line.split(":")
        name = words[0]
        feature_name.append(name)

data = pd.DataFrame(data.values, columns=[feature_name])
data.to_csv("data.csv", index=False)
