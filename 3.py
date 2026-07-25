from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

data=load_iris()

X_train,X_test,y_train,y_test=train_test_split(
data.data,data.target,test_size=0.2,random_state=42)

model=DecisionTreeClassifier()

model.fit(X_train,y_train)

pred=model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,pred))
print("Confusion Matrix")
print(confusion_matrix(y_test,pred))
