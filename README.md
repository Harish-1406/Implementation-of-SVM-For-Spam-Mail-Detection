# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## AlgorithmStart the program
1.Import the python pandas library as pd
2.Read the contents of the Spam csv file
3.Display the first 5 rows of the dataset using head()
4.Assign x as v1 values and y as v2 values
5.From sklearn library select the feature extraction and import CountVectorizer
6.CountVectorizer will convert the Text to Numerical Data
7.From sklearn library import Support Vector Classifier (ie. SVC)
8.Predict the x_test using SVC
9.Print the accuracy of the SVM Model 

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: Harish P K
RegisterNumber:  212224040104

```
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding="windows-1252")
data.head()
data.info()
data.shape
x=data['v1'].values
y=data['v2'].values
x.shape
y.shape
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
x_train.shape
x_test.shape
y_train.shape
y_test.shape
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test=cv.transform(x_test)
x_train.shape
x_test.shape
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![image](https://github.com/user-attachments/assets/aa21a522-daf1-4370-9b9e-a74d0e6de5ff)
![image](https://github.com/user-attachments/assets/85280531-7ff8-4b20-b7ba-ee8c1d725bda)
![image](https://github.com/user-attachments/assets/eee09fd7-5da7-4d40-a0cc-0028b92a4615)
![image](https://github.com/user-attachments/assets/51b81f63-4a63-455a-b521-faf97a730001)
![image](https://github.com/user-attachments/assets/33b89ee9-872f-4faa-b33e-7972121fb61f)
![image](https://github.com/user-attachments/assets/425877be-8233-4417-89d3-123c39d7b3b0)
![image](https://github.com/user-attachments/assets/e7cdbbcd-e7e8-4137-87bc-2fa406bb801a)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
