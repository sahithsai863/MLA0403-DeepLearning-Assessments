from sklearn.linear_model import SGDRegressor
from sklearn.datasets import make_regression

X,y=make_regression(n_samples=100,n_features=1)

model=SGDRegressor(max_iter=1000)

model.fit(X,y)

print("Coefficient:",model.coef_)
