import pandas as pd 
stroke = pd.read_csv('healthcare-dataset-stroke-data.csv')
#Handle missing values
stroke.drop(["smoking_status"], axis = 1 , inplace = True)
#Convert non-numeric data to numeric
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
stroke['gender']=le.fit_transform(stroke['gender'])
stroke['ever_married']=le.fit_transform(stroke['ever_married'])
stroke['work_type']=le.fit_transform(stroke['work_type'])
stroke['Residence_type']=le.fit_transform(stroke['Residence_type'])
stroke

test_data = stroke[stroke["bmi"].isnull()]
test_data

stroke.dropna(inplace=True)
stroke

y_train = stroke["bmi"]
y_train

x_train = stroke.drop("bmi", axis=1)
x_train

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train , y_train)

X_test = test_data.drop("bmi", axis=1)
X_test

y_pred = lr.predict(X_test)
pd.DataFrame(y_pred)

test_data.loc[test_data.bmi.isnull(), 'bmi'] = y_pred
test_data

stroke = pd.read_csv('healthcare-dataset-stroke-data.csv')
stroke.drop(["smoking_status"], axis = 1 , inplace = True)
stroke

stroke.loc[stroke.bmi.isnull(), 'bmi'] = test_data
bmi=stroke['bmi']
import pandas as pd 
import numpy as np
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
test_data = stroke[stroke["smoking_status"].isnull()]
stroke.dropna(inplace=True)
y_train = stroke["smoking_status"]
x_train = stroke.drop("smoking_status", axis=1)
from sklearn.neighbors import KNeighborsClassifier
#Create KNN classifier
model = KNeighborsClassifier(n_neighbors=20)
# Train the model using the training sets
model.fit(x_train,y_train)
X_test = test_data.drop("smoking_status", axis=1)
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
smoking_status=stroke['smoking_status']
stroke = pd.read_csv('healthcare-dataset-stroke-data.csv')
stroke.loc[stroke['smoking_status']=='Unknown','smoking_status'] = np.nan
stroke.loc[stroke.bmi.isnull(), 'bmi'] = bmi
stroke.loc[stroke.smoking_status.isnull(), 'smoking_status'] = smoking_status
stroke['gender']=le.fit_transform(stroke['gender'])
stroke['ever_married']=le.fit_transform(stroke['ever_married'])
stroke['work_type']=le.fit_transform(stroke['work_type'])
stroke['Residence_type']=le.fit_transform(stroke['Residence_type'])
stroke['smoking_status']=le.fit_transform(stroke['smoking_status'])
stroke




#Check outliers
low=0.05
high=0.95
qnt=stroke.quantile([low,high])
#Replace outliers with mean
stroke.age = stroke.age.apply(lambda v: v if qnt.age[low]< v <qnt.age[high] else stroke['age'].mean())
stroke.bmi = stroke.bmi.apply(lambda v: v if qnt.bmi[low]< v <qnt.bmi[high] else stroke['bmi'].mean())
stroke.avg_glucose_level = stroke.avg_glucose_level.apply(lambda v: v if qnt.avg_glucose_level[low]< v <qnt.avg_glucose_level[high] else stroke['avg_glucose_level'].mean())
stroke.describe()




#Check the corrilation 
stroke.corr()["stroke"].sort_values(ascending=False)




#Drop iirrelevant features
stroke.drop(["work_type"], axis = 1 , inplace = True)
stroke.drop(["gender"], axis = 1 , inplace = True)
stroke.drop(["Residence_type"], axis = 1 , inplace = True)




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
