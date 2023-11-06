#Example 3.3 Python SVM Iris CSV Classifications
from sklearn import svm, datasets
import pandas as pd

#df = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv') 
#df.to_csv('./iris.csv', index=False)
df = pd.read_csv('iris.csv')
#pd.set_option('display.max_rows', None)
print(df)
X = df.values[:,:2]
s = df['species']
d = dict([(y,x) for x,y in enumerate(sorted(set(s)))])
y = [d[x] for x in s]
 
clf = svm.SVC()
clf.fit(X, y)
#Predict the flower for a given Sepal length and width
p = clf.predict([[5.4, 3.2]])
print(p) 