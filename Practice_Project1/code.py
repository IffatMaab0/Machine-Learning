import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error , mean_squared_error
import numpy as np

data = pd.read_csv('student.csv')
X = data[['Hours']]
Y = data['Score']

model = LinearRegression()
model.fit(X, Y)
predicted_score = model.predict(X)

mae = mean_absolute_error(Y, predicted_score)
mse = mean_squared_error(Y, predicted_score)
rmse = np.sqrt(mse)

print("Mean Absolute Error: ",mae)
print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ", rmse)
new_pred = model.predict([[7]])
print('New Score Prediction', new_pred)