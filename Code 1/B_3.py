#Libraries
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

#Read CSV
stroke = pd.read_csv('healthcare-dataset-stroke-data.csv')

#Drop smoking status column
stroke.drop(["smoking_status"], axis = 1 , inplace = True)

#Transform string data into integer data
le=LabelEncoder()
stroke['gender']=le.fit_transform(stroke['gender'])
stroke['ever_married']=le.fit_transform(stroke['ever_married'])
stroke['work_type']=le.fit_transform(stroke['work_type'])
stroke['Residence_type']=le.fit_transform(stroke['Residence_type'])
stroke

#Contains missing values - Test data
test_data = stroke[stroke["bmi"].isnull()]
test_data

#Drop missing values from dataframe - Training data
stroke.dropna(inplace=True)
stroke

y_train = stroke["bmi"]
y_train


x_train = stroke.drop("bmi", axis=1)
x_train


#Create linear regression model and train it 
lr = LinearRegression()
lr.fit(x_train , y_train)


X_test = test_data.drop("bmi", axis=1)
X_test


#Make prediction for bmi
y_pred = lr.predict(X_test)
pd.DataFrame(y_pred)


test_data.loc[test_data.bmi.isnull(), 'bmi'] = y_pred
test_data

stroke = pd.read_csv('healthcare-dataset-stroke-data.csv')
stroke.drop(["smoking_status"], axis = 1 , inplace = True)
stroke

#Insert predicted bmi values to dataframe
stroke.loc[stroke.bmi.isnull(), 'bmi'] = test_data
stroke
