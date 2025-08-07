import pandas as pd
import sklearn.linear_model as ln
mydata=pd.read_csv("weight_predict_real.csv")
x=mydata[["height","age","bmi","muscle_mass","body_fat"]]
y=mydata[["weight"]]
model=ln.LinearRegression()
model.fit(x,y)
print("coeffient:",model.coef_)
print("intercept",model.intercept_)
print(model.predict([[161.48769937914997,61,22.658363281836316,33.32720895136154,25.669645543580984]]))