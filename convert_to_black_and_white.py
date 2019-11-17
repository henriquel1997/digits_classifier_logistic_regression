import pandas as pd

X = pd.read_csv("mnist_test.csv")
labels = X["label"].to_list()

del X["label"]

for name in X.columns:
    X[name] = X[name].mask(X[name] > 0, 255)

X.insert(0, "label", labels)
print(X)

X.to_csv("mnist_test_new.csv", index=False)
