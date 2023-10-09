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
model = KNeighborsClassifier(n_neighbors=10)
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
stroke['gender']=le.fit_transform(stroke['gender'])
stroke['ever_married']=le.fit_transform(stroke['ever_married'])
stroke['work_type']=le.fit_transform(stroke['work_type'])
stroke['Residence_type']=le.fit_transform(stroke['Residence_type'])
stroke['smoking_status']=le.fit_transform(stroke['smoking_status'])
stroke
#Check the corrilation 
stroke.corr()["stroke"].sort_values(ascending=False)




#Define dependent variable(stroke)
Y = stroke['stroke'].values
#Drop irrelevant columns(id) and Define independent variables
X = stroke.drop(labels=['id' , 'stroke'] , axis=1)
#Split data into train and test datasets(75%-25%)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=20)
#Run the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
#Generate model 
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
