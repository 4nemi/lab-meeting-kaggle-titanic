import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def load_data(features):
    dfs = [pd.read_feather(f"./features/{f}_train.ftr") for f in features]
    X_train = pd.concat(dfs, axis=1)
    dfs = [pd.read_feather(f"./features/{f}_test.ftr") for f in features]
    X_test = pd.concat(dfs, axis=1)
    return X_train, X_test


def train():
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")
    train_feat, test_feat = load_data(["Age"])

    y = train["Survived"]

    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(train[features])
    X_test = pd.get_dummies(test[features])

    X = pd.concat([X, train_feat], axis=1)
    X_test = pd.concat([X_test, test_feat], axis=1)

    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=3)
    model.fit(X, y)
    predictions = model.predict(X_test)

    output = pd.DataFrame({"PassengerId": test.PassengerId, "Survived": predictions})
    output.to_csv("../output/submission.csv", index=False)


if __name__ == "__main__":
    train()
