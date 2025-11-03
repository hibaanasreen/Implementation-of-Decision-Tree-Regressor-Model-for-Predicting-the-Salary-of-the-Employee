# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.Calculate Mean square error,data prediction and r2.
## Program:
```

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Hiba Nasreen M
RegisterNumber:212224040117  

```
```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_csv('Salary.csv')

print(dataset.head())

X = dataset[['Level']]  
y = dataset['Salary']            

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = DecisionTreeRegressor(random_state=42)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

level = 6.5
predicted_salary = regressor.predict(pd.DataFrame([[Level]], columns=['Level']))
print(f"Predicted salary for {level} years of experience is: {predicted_salary[0]}")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

X_grid = np.arange(min(X.values), max(X.values), 0.01) 
X_grid = X_grid.reshape(-1, 1)

plt.scatter(X, y, color='red')                
plt.plot(X_grid, regressor.predict(X_grid), color='blue') 
plt.title('Decision Tree Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
```



## Output:
<img width="306" height="235" alt="image" src="https://github.com/user-attachments/assets/547828f5-d535-4e6d-ae58-11737eadac5d" />

<img width="770" height="259" alt="image" src="https://github.com/user-attachments/assets/387dfd6f-22b3-4250-baca-e0708bac48db" />

<img width="893" height="104" alt="image" src="https://github.com/user-attachments/assets/0c618816-d856-4419-b063-2ce592f04076" />

<img width="784" height="233" alt="image" src="https://github.com/user-attachments/assets/f67416bf-1331-490d-ba2a-cfece0b53cc2" />

<img width="635" height="42" alt="image" src="https://github.com/user-attachments/assets/d0d6bb17-1f05-4a4b-a4ca-30c8ed22a141" />

<img width="600" height="43" alt="image" src="https://github.com/user-attachments/assets/4e351bd5-d336-4f28-a816-801ce16be5b7" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
