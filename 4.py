from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

iris=load_iris()

X_train,X_test,y_train,y_test=train_test_split(
iris.data,iris.target,test_size=0.2,random_state=1)

model=MLPClassifier(hidden_layer_sizes=(10,),max_iter=1000)

model.fit(X_train,y_train)

print("Accuracy:",model.score(X_test,y_test))
