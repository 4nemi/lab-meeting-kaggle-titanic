import pandas as pd

from base import Feature, get_arguments, generate_features

Feature.dir = "."


class Pclass(Feature):
    def create_features(self):
        # create one-hot encoding features
        self.train["Pclass"] = train["Pclass"].astype(str)
        self.test["Pclass"] = test["Pclass"].astype(str)
        self.train = pd.get_dummies(self.train, columns=["Pclass"])
        self.test = pd.get_dummies(self.test, columns=["Pclass"])


if __name__ == "__main__":
    args = get_arguments()

    train = pd.read_csv("../../data/train.csv")
    test = pd.read_csv("../../data/test.csv")

    generate_features(globals(), args.force)
