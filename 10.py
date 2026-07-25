from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

for dim in [2,5,10,20]:

    X,y=make_classification(
        n_samples=500,
        n_features=dim,
        n_informative=2,
        n_redundant=0,
        random_state=42
    )

    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2)

    model=KNeighborsClassifier()

    model.fit(X_train,y_train)

    print("Dimension:",dim,
          "Accuracy:",model.score(X_test,y_test))
