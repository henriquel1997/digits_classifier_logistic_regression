from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle

X = pd.read_csv("mnist_train_new.csv")
labels = X["label"].to_list()

del X["label"]

print(X)
print(labels)

print("Treinando")
logReg = LogisticRegression(multi_class='ovr', solver='lbfgs', random_state=0).fit(X, labels)
print("Fim do Treinamento")

with open('mnist_digits_ovr_bw.joblib', 'wb') as file:
    pickle.dump(logReg, file)
