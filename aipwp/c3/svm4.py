#Example 3.4 Python SVM Iris URL Classifications
from sklearn import svm, datasets
import pandas as pd
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
 
df = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
 
print(df.shape)
print(df.head(10))
print(df.tail(10))
print(df.describe())
 
#count NAN
print("number of nan" + str(df.isna().sum().sum()))
#drop NAN values
df = df.dropna()
 
print(df.groupby('species').size())
# histograms
df.hist()
pyplot.show()
# scatter plot matrix
scatter_matrix(df)
pyplot.show()
 
X = df.values[:,:2]
s = df['species']
d = dict([(y,x) for x,y in enumerate(sorted(set(s)))])
y = [d[x] for x in s]
 
clf = svm.SVC()
clf.fit(X, y)
#Predict the flower for a given Sepal length and width
p = clf.predict([[5.4, 3.2]])
print(p) 