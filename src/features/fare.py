import pandas as pd

from base import Feature, get_arguments, generate_features

Feature.dir = "."


class Fare(Feature):
    def create_features(self):
        self.train["Fare"] = train["Fare"]
        self.test["Fare"] = test["Fare"]
        for data in [self.train, self.test]:
            data["Fare"].fillna(data["Fare"].median(), inplace=True)
            data["Fare"] = data["Fare"].astype(float)


if __name__ == "__main__":
    args = get_arguments()

    train = pd.read_csv("../../data/train.csv")
    test = pd.read_csv("../../data/test.csv")

    generate_features(globals(), args.force)
