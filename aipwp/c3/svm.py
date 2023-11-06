#Example 3.1 Python SVM Classifications
# a simple SVM gender classification example based on height, 
# weight, and shoe size. It first uses from sklearn import svm to import the SVM library. 
# It uses an array called X to store four sets of values of height in centimeters, weight in kilos, 
# and shoe size in UK size, and uses an array named y to store four sets of known genders, 0 for Male, 1 for Female. 
# It then trains the SVM classifier and makes a prediction for a given height, weight, and shoe size [[160, 60, 7]].
 
from sklearn import svm
X = [[170, 70, 10], [180, 80,12], [170, 65, 8],[160, 55, 7],[177, 65, 9], [177,86,10]] #Height[cm], Weight[kg], Shoesize[UK]
y = [0, 0, 1, 1, 1, 0]    #Gender, 0: Male, 1: Female
clf = svm.SVC()
clf.fit(X, y)
#Predict 
p = clf.predict([[160, 75, 7]])
print(p) 
p = clf.predict([[177, 86, 10]])
print(p) 