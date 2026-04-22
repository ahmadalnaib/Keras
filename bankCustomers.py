import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler 
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('Churn_Modelling.csv')
print(df.head())

value_counts=df['Exited'].value_counts()
print(value_counts)

labels=value_counts.index
counts=value_counts.values

plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Exited Customers')
# plt.show()

input_columns=df.drop('Exited', axis=1)
target_column=df['Exited']

oversampler=RandomOverSampler(random_state=0)
input_columns_resampled, target_column_resampled=oversampler.fit_resample(input_columns, target_column)
df_balanced=pd.concat([input_columns_resampled, target_column_resampled], axis=1)
class_distribution=df_balanced['Exited'].value_counts()
print(class_distribution)

x=df_balanced.iloc[:, 3:13].values
print(x)

y=df_balanced.iloc[:, 13].values
print(y)

labelencoder_gender=LabelEncoder()
x[:, 2]=labelencoder_gender.fit_transform(x[:, 2])
print(x[:,2])