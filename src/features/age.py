import pandas as pd
import numpy as np

from base import Feature, get_arguments, generate_features

Feature.dir = "."


class Age(Feature):
    def create_features(self):
        self.train["Age"] = train["Age"]
        self.test["Age"] = test["Age"]
        for data in [self.train, self.test]:
            mean = train["Age"].mean()
            std = test["Age"].std()
            is_null = data["Age"].isnull().sum()
            rand_age = np.random.randint(mean - std, mean + std, size=is_null)

            age_slice = data["Age"].copy()
            age_slice[np.isnan(age_slice)] = rand_age
            data["Age"] = age_slice
            data["Age"] = data["Age"].astype(int)


if __name__ == "__main__":
    args = get_arguments()

    train = pd.read_csv("../../data/train.csv")
    test = pd.read_csv("../../data/test.csv")

    generate_features(globals(), args.force)
