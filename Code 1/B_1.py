#Libraries
import pandas as pd
import numpy as np

#Read CSV
stroke = pd.read_csv('healthcare-dataset-stroke-data.csv')

#See the missing values in smoking status
stroke[stroke["smoking_status"] == 'Unknown']

#See the missing values of bmi
stroke.isnull().sum()

#Drop bmi column
stroke.drop(["bmi"], axis = 1 , inplace = True)
stroke

#Drop smoking status column
stroke.drop(["smoking_status"], axis = 1 , inplace = True)
stroke
