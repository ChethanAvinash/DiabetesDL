import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense

diabetes = pd.read_csv('diabetes.csv')
cols_to_norm = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
       'DiabetesPedigreeFunction', 'Age']
dia_norm = diabetes.copy()
dia_norm[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min())/ (x.max() - x.min()) )
x = dia_norm.drop(columns = 'Outcome')
y = dia_norm['Outcome']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=500, batch_size=100)
_, accuracy = model.evaluate(X_train, y_train)
predictions = model.predict(X_test)
rounded = [round(x[0]) for x in predictions]
pred = np.array(rounded)
count = 0
for i,j in zip(pred,y_test):
  if i==j:
    count+=1
print(count/y_test.size)
model.save('diabetes-model.h5')
