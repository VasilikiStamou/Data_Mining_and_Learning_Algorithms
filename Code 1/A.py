#Libraries
import pandas as pd
import seaborn as sns

sns.set(color_codes=True)

stroke = pd.read_csv('healthcare-dataset-stroke-data.csv')

stroke.head()
stroke.info()

#Plots
sns.jointplot(stroke['age'],stroke['stroke'])
sns.jointplot(stroke['gender'],stroke['stroke'])
sns.jointplot(stroke['work_type'],stroke['stroke'])
sns.jointplot(stroke['hypertension'],stroke['stroke'])
sns.jointplot(stroke['heart_disease'],stroke['stroke'])
sns.jointplot(stroke['ever_married'],stroke['stroke'])
sns.jointplot(stroke['Residence_type'],stroke['stroke'])
sns.jointplot(stroke['smoking_status'],stroke['stroke'])
sns.jointplot(stroke['avg_glucose_level'],stroke['stroke'])
sns.jointplot(stroke['bmi'],stroke['stroke'])
