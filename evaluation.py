import copy
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def likelihood(y, y_pred):
    error = y - y_pred
    v = error.var()
    return np.prod(1 / np.sqrt(2*np.pi*v)*np.exp((error**2)/(2*v)))

def log_likelihood(y, y_pred):
    error = y - y_pred
    v = error.var()
    N = len(error)
    return -N / 2 * np.log(2*np.pi*v) - 1 /(2*v)*np.sum(error**2)

def R_score(y, y_pred):
    y_mean = y.mean()
    return 1 - ((y - y_pred)**2).sum() / ((y - y_mean)**2).sum()

def AIC(y, y_pred, X):
    params = X.shape[1]
    L =  log_likelihood(y, y_pred)
    return -2 * L + 2 * (params+1)

def BIC(y, y_pred, X):
    N, params = X.shape
    L =  log_likelihood(y, y_pred)
    return -2 * L + (params+1) * np.log(N)

class Stepwise():

    def __init__(self, X_train, y_train, mode="FWBW", evaluation="AIC"):
        self.mode = mode
        self.lr = LinearRegression()
        self.scores, self.evals, self.errors, self.preds, self.idx, self.coef, self.intercept = [], [], [], [], [], [],[]
        self.params_num, self.eval_best = 0, 0
        self.index_best = []
        self.X_train, self.y_train = np.array(X_train), np.array(y_train)
        if evaluation == "BIC":
            self.eval = BIC
        else:
            self.eval = AIC

    def run(self):
        indexs = self.run_1st()
        if self.mode=="FWBW":
            self.stepwiseFWBW(indexs)
        else:
            self.stepwiseFW(indexs)

    def run_1st(self):
        self.params_num = self.X_train.shape[1]
        for i in range(self.params_num):
            x = self.X_train[:,i].reshape(-1,1)
            self.core(x, self.y_train)
            self.idx.append([i])

        self.index_best.append(np.array(self.scores).argmax())
        self.eval_best = self.evals[self.index_best[0]]
        indexs = list(np.argsort(-np.array(self.scores)))
        self.sort(indexs)
        indexs = indexs[1:]
        return indexs

    def stepwiseFWBW(self, indexs):
        flag = True
        while flag:
            flag= False
            j_best_temp, jdx_best_temp = 0, 0
            for j, jdx in enumerate(indexs):
                self.index_best.append(jdx)
                self.core(self.X_train[:,self.index_best], self.y_train)
                self.idx.append( copy.deepcopy(self.index_best))
                del self.index_best[-1]
                if self.evals[-1] >= self.eval_best:
                    pass
                else:
                    self.eval_best = self.evals[-1]
                    j_best_temp = j
                    jdx_best_temp = jdx
                    flag = True # このfor loopでパラメータを見つけたため、while継続

            if flag:
                del indexs[j_best_temp]
                self.index_best.append(jdx_best_temp)


    def stepwiseFW(self, indexs):
        flag = True
        while flag:
            flag= False
            for j, jdx in enumerate(indexs):
                self.index_best.append(jdx)
                self.core(self.X_train[:,self.index_best], self.y_train)
                self.idx.append( copy.deepcopy(self.index_best))
                if self.evals[-1] >= self.eval_best:
                    del self.index_best[-1]
                else:
                    self.eval_best = self.evals[-1]
                    del indexs[j]
                    flag = True # このfor loopでパラメータを見つけたため、while継続
                    break

    def core(self, X, y):
        XT = self.translateRank(X)
        fit = self.lr.fit(XT, y)
        y_pred = fit.predict(XT)
        self.evals.append(self.eval(y, y_pred, XT))
#         self.evals.append(AIC(y, y_pred, XT))
        self.preds.append(y_pred)
        self.scores.append(fit.score(XT, y))
        self.coef.append(fit.coef_)
        self.intercept.append(fit.intercept_)

    def mse(self, X, y):
        X = np.array(X)
        y = np.array(y)
        for i, co, inter in zip(self.idx, self.coef, self.intercept):
            y_pred = np.dot(X[:, i], co) + inter
            self.errors.append(mean_squared_error(y, y_pred))

    def translateRank(self, x):
        if x.ndim == 1:
            return x.reshape(-1,1)
        else:
            return x

    def sort(self, idx):
        self.evals = self.listGetter(self.evals, idx)
        self.preds = self.listGetter(self.preds, idx)
        self.scores = self.listGetter(self.scores, idx)
#         self.errors = listGetter(self.errors, idx)
        self.coef = self.listGetter(self.coef, idx)
        self.intercept = self.listGetter(self.intercept, idx)
        self.idx = self.listGetter(self.idx, idx)

    def listGetter(self, l, idx):
        return [l[i] for i in idx  ]
