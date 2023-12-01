from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets

iris = datasets.load_iris()
x, y = iris.data, iris.target
print('Features: sepal-length, sepal-width, petal-length, petal-width')
print(x)
print('Classes: 0-Iris-Setosa, 1-Iris-Versicolour, 2-Iris-Virginica')
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nAccuracy Metrics:')
print(classification_report(y_test, y_pred))
