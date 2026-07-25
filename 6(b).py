from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris=load_iris()

X_train,X_test,y_train,y_test=train_test_split(
iris.data,iris.target,test_size=0.2)

p=Perceptron()
p.fit(X_train,y_train)

mlp=MLPClassifier(hidden_layer_sizes=(20,),max_iter=1000)
mlp.fit(X_train,y_train)

print("Perceptron:",p.score(X_test,y_test))
print("MLP:",mlp.score(X_test,y_test))
