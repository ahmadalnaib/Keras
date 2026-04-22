import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler 
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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

labelencoder_gender.transform(['Female', 'Male'])
# array([0, 1])

distinct_values=np.unique(x[:, 1])
print(distinct_values)
column_transformer=ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x=column_transformer.fit_transform(x)
print(x)

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=0)


sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
print(x_train)

