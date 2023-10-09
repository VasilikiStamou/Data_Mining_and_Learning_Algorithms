import pandas as pd
import numpy as np
stroke = pd.read_csv('healthcare-dataset-stroke-data.csv')
#Handle missing values
stroke['bmi'].fillna(float(stroke['bmi'].mean()), inplace=True)
stroke.drop(["smoking_status"], axis = 1 , inplace = True)
#Convert non-numeric data to numeric
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
stroke['gender']=le.fit_transform(stroke['gender'])
stroke['ever_married']=le.fit_transform(stroke['ever_married'])
stroke['work_type']=le.fit_transform(stroke['work_type'])
stroke['Residence_type']=le.fit_transform(stroke['Residence_type'])
stroke.describe()

#Check outliers
low=0.04
high=0.96
qnt=stroke.quantile([low,high])
#Replace outliers with mean
stroke.age = stroke.age.apply(lambda v: v if qnt.age[low]< v <qnt.age[high] else stroke['age'].mean())
stroke.avg_glucose_level = stroke.avg_glucose_level.apply(lambda v: v if qnt.avg_glucose_level[low]< v <qnt.avg_glucose_level[high] else stroke['avg_glucose_level'].mean())
stroke.bmi = stroke.bmi.apply(lambda v: v if qnt.bmi[low]< v <qnt.bmi[high] else stroke['bmi'].mean())
stroke.describe()

#Check the corrilation 
stroke.corr()["stroke"].sort_values(ascending=False)

#Drop iirrelevant features
stroke.drop(["work_type"], axis = 1 , inplace = True)
stroke.drop(["gender"], axis = 1 , inplace = True)

#Normalize feautures
stroke['age']=stroke['age'].apply(lambda v:(v-stroke['age'].min())/(stroke['age'].max()-stroke['age'].min()))
stroke['avg_glucose_level']=stroke['avg_glucose_level'].apply(lambda v:(v-stroke['avg_glucose_level'].min())/(stroke['avg_glucose_level'].max()-stroke['avg_glucose_level'].min()))
stroke['bmi']=stroke['bmi'].apply(lambda v:(v-stroke['bmi'].min())/(stroke['bmi'].max()-stroke['bmi'].min()))
stroke


#Define dependent variable(stroke)
Y = stroke['stroke'].values
#Drop irrelevant columns(id) and Define independent variables
X = stroke.drop(labels=['id' , 'stroke'] , axis=1)
#Split data into train and test datasets(75%-25%)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=20)
#Run the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 10, random_state=30)
#Fit data with model
model.fit(X_train,Y_train)
#Predict the result using the model with X_test
prediction_test = model.predict(X_test)
#Compare prediction_test - Y_test statistically
from sklearn.metrics import f1_score,precision_score,recall_score,classification_report
F1 = f1_score(Y_test,prediction_test,average="macro")
Precision = precision_score(Y_test,prediction_test,average="macro")
Recall = recall_score(Y_test,prediction_test,average="macro")
Classification_Report = classification_report(Y_test,prediction_test)
#Check which parameters are contributing the best
feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
print(feature_imp)

print(Classification_Report)
