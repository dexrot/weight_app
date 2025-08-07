import pandas as pd
import sklearn.linear_model as lm
from sklearn.preprocessing import LabelEncoder
mydata=pd.read_csv("big.csv")
ge_le =LabelEncoder()
bo_le =LabelEncoder()
mydata["Gender_encoded"]=ge_le.fit_transform(mydata[["Gender"]])
mydata["BodyType_encoded"]=bo_le.fit_transform(mydata[["Body Type"]])
x=mydata[["Age","Gender_encoded","BodyType_encoded","Height"]]
y=mydata[["Weight"]]
model=lm.LinearRegression()
model.fit(x,y)
print("coefficent:",model.coef_[0])
print("intercept",model.intercept_[0])
Age=int(input("enter your age ="))
Gender=input("enter your gender(use  capital M/G  as starting) =")
BodyType=input("enter your body type=")
Height=int(input("enter your height ="))
your_weight =model.predict([[Age,ge_le.transform([Gender])[0],bo_le.transform([BodyType])[0],Height]])
print("your weight approximatly",your_weight)