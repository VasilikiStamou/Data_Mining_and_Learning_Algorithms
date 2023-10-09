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
#Handle missing values
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
#Place the predicted values into the dataset
stroke = pd.read_csv('healthcare-dataset-stroke-data.csv')
stroke.drop(["smoking_status"], axis = 1 , inplace = True)
stroke

stroke.loc[stroke.bmi.isnull(), 'bmi'] = test_data
stroke
#Convert non-numeric data to numeric
le=LabelEncoder()
stroke['gender']=le.fit_transform(stroke['gender'])
stroke['ever_married']=le.fit_transform(stroke['ever_married'])
stroke['work_type']=le.fit_transform(stroke['work_type'])
stroke['Residence_type']=le.fit_transform(stroke['Residence_type'])
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
