Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
##AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

##EQUIPMENT REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook

##ALGORITHM:
Gather data consisting of two variables. Input- a factor that affects the marks and Output - the marks scored by students
Plot the data points on a graph where x-axis represents the input variable and y-axis represents the marks scored
Define and initialize the parameters for regression model: slope controls the steepness and intercept represents where the line crsses the y-axis
Use the linear equation to predict marks based on the input Predicted Marks = m.(hours studied) + b
for each data point calculate the difference between the actual and predicted marks
Adjust the values of m and b to reduce the overall error. The gradient descent algorithm helps update these parameters based on the calculated error
Once the model parameters are optimized, use the final equation to predict marks for any new input data

##PROGRAM:
Program to implement the simple linear regression model for predicting the marks scored.
```
Developed by: SHAHIN J
RegisterNumber: 212223040190
```
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#read csv file
df=pd.read_csv('/content/student_scores (1).csv')

#displaying the content in datafile
print(df.head())
print(df.tail())

# Segregating data to variables
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,-1].values
print(y)

print(x.shape)
print(y.shape)

#splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#import linear regression model and fit the model with the data
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)

#displaying predicted values
y_pred=reg.predict(x_test)
x_pred=reg.predict(x_train)
print(y_pred)
print(x_pred)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#find mae,mse,rmse
mse = mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae = mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse = np.sqrt(mse)
print('RMSE = ',rmse)

#graph plot
plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,y_pred,color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

y_pred1=reg.predict(np.array([[13]]))
y_pred1
```
##OUTPUT

##Head Values
![image](https://github.com/user-attachments/assets/d4dc5599-1809-41d8-8a0c-1dfd7af52193)
##Tail Values
![image](https://github.com/user-attachments/assets/d761f299-963c-437f-801b-33130533106c)
##X Values
![image](https://github.com/user-attachments/assets/452ffcfe-258f-4ee3-9c20-a4f12c44494c)
##Y Values
![image](https://github.com/user-attachments/assets/e0cbb7e4-3ee2-45bd-b911-35482748ab44)
##Shape Values
![image](https://github.com/user-attachments/assets/a1df5e9a-f329-44ae-ad12-ec956d71b0f7)
##Predicted Values
![image](https://github.com/user-attachments/assets/ff86528a-9d81-4f20-904d-45f081d63811)
##Training and Testing Shapes
![image](https://github.com/user-attachments/assets/6f64e7bc-e12a-4f72-adf5-881ce5bc6c56)
##MAE,MSE,RMSE
![image](https://github.com/user-attachments/assets/a5ed64d0-c172-4800-93e4-bae44122562a)
##Graph Plot
![image](https://github.com/user-attachments/assets/e63c48d9-315a-41e2-92ee-03027164b0a1)
##Array Values
![image](https://github.com/user-attachments/assets/be1d0746-62e1-4fc8-8d8e-dba2bf5a65cf)

##Result
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
