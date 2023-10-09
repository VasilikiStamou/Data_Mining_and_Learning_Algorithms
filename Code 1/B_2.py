#Libraries
import pandas as pd

#Read CSV
stroke = pd.read_csv('healthcare-dataset-stroke-data.csv')

#Fill missing values of bmi with the mean of bmi non-missing values
stroke['bmi'].fillna(float(stroke['bmi'].mean()), inplace=True)

#Drop smoking status column
stroke.drop(["smoking_status"], axis = 1 , inplace = True)

#See the final dataframe
stroke




