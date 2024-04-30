import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
df=pd.read_csv("diabetes.csv")
print(df.head())

X = df.drop(columns = 'Outcome', axis=1)
Y = df['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

from sklearn import svm
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

from sklearn.metrics import accuracy_score
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)
import pickle

with open('modeln.pkl', 'wb') as m:
    pickle.dump(classifier, m)

with open('modeln.pkl', 'rb') as m:
    model = pickle.load(m)

    

