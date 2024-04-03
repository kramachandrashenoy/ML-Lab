#using sklearn
import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

iris = pd.read_csv(r"C:\Users\Ramachandra\OneDrive\Desktop\ML Lab\Iris.csv")
iris.head()

X = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print('accuracy is', accuracy_score(y_test, y_pred))
