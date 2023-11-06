#Example 3.2 Python SVM Iris Classifications
from sklearn import svm, datasets
iris = datasets.load_iris()
# Take the first two features: Sepal length and Sepal width
X = iris.data[:, 2:4]
y = iris.target #0: Setosa, 1: Versicolour, 2:Virginica
print(y)
clf = svm.SVC()
clf.fit(X, y)
#Predict the flower for a given Sepal length and width
p = clf.predict([[5.4, 3.2]])
print(p) 