import pandas as pd

from base import Feature, get_arguments, generate_features

Feature.dir = "."


class Embarked(Feature):
    def create_features(self):
        # create one-hot encoding features
        self.train["Embarked"] = train["Embarked"].astype(str)
        self.test["Embarked"] = test["Embarked"].astype(str)
        self.train = pd.get_dummies(self.train, columns=["Embarked"])
        self.train.drop(["Embarked_nan"], axis=1, inplace=True)
        self.test = pd.get_dummies(self.test, columns=["Embarked"])


if __name__ == "__main__":
    args = get_arguments()

    train = pd.read_csv("../../data/train.csv")
    test = pd.read_csv("../../data/test.csv")

    generate_features(globals(), args.force)
