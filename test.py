import pickle
import pandas as pd


def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


test = pd.read_csv("mnist_test_new.csv")
test_labels = test["label"].to_list()
del test["label"]

logReg = load_model('mnist_digits_multi_bw.joblib')

print("Score: \n", logReg.score(test, test_labels))
print("Parametros: \n", logReg.get_params())
print("Coeficientes: \n", logReg.coef_)
print("Intercepts: \n", logReg.intercept_)
print("Número de iterações: \n", logReg.n_iter_)
