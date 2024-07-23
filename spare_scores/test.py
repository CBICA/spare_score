import pandas as pd

from .mlp_torch import MLPTorchModel

if __name__ == "__main__":
    data = pd.read_csv("../../moonDataset.csv")
    predictors = ["X1", "X2", "X3"]
    to_predict = "label"

    model = MLPTorchModel(predictors, to_predict, "")
    model.fit(data)
