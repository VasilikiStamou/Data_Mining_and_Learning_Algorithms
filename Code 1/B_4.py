#Libraries
import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

#Read CSV
stroke = pd.read_csv('healthcare-dataset-stroke-data.csv')

#Handle missing values
stroke.loc[stroke['smoking_status']=='Unknown','smoking_status'] = np.nan
stroke.drop(["bmi"], axis = 1 , inplace = True)

#Convert non-numeric data to numeric
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
stroke['gender']=le.fit_transform(stroke['gender'])
stroke['ever_married']=le.fit_transform(stroke['ever_married'])
stroke['work_type']=le.fit_transform(stroke['work_type'])
stroke['Residence_type']=le.fit_transform(stroke['Residence_type'])
def smoking_status_to_num(x):
    if x == 'formerly smoked':
        return 0
    if x == 'never smoked':
        return 1
    if x == 'smokes':
        return 2
    
    
stroke['smoking_status'] = stroke['smoking_status'].apply(smoking_status_to_num)

#Test data
test_data = stroke[stroke["smoking_status"].isnull()]
#Train data
stroke.dropna(inplace=True)
y_train = stroke["smoking_status"]
x_train = stroke.drop("smoking_status", axis=1)

#Create KNN classifier
model = KNeighborsClassifier(n_neighbors=10)

# Train the model using the training sets
model.fit(x_train,y_train)

X_test = test_data.drop("smoking_status", axis=1)
#Make prediction
y_pred = model.predict(X_test)

test_data.loc[test_data.smoking_status.isnull(), 'smoking_status'] = y_pred
test_data

def smoking_status_to_cat(x):
        if x == 0:
            return 'formerly smoked'
        if x == 1:
            return 'never smoked'
        if x == 2 :
            return 'smokes'
        
test_data['smoking_status'] = test_data['smoking_status'].apply(smoking_status_to_cat)
test_data

stroke = pd.read_csv('healthcare-dataset-stroke-data.csv')
stroke.drop(["bmi"], axis = 1 , inplace = True)
stroke.loc[stroke['smoking_status']=='Unknown','smoking_status'] = np.nan

stroke.loc[stroke.smoking_status.isnull(), 'smoking_status'] = test_data
stroke
